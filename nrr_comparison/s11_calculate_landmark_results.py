import numpy as np
import os
import pickle
from os.path import join
from utils import getFiles, getDirs
import open3d as o3d
from scipy.spatial import KDTree
from tqdm import tqdm


## functions to test correspondence and interpolation
def calculate_click_error(sequence_fnames, results_folder, structure_name, click_directory, click_name):
    mesh_error_x = []
    mesh_error_y = []
    corr_error_x = []
    corr_error_y = []
    reg_error = []
    for sequence_fname in tqdm(sequence_fnames):
        with open(join(results_folder, sequence_fname), "rb") as f:
            result = pickle.load(f)

        assignment = result["assignment"]
        assignmentinv = result["assignmentinv"]
        verts_x, triangles_x = result["X"]["verts"], result["X"]["triangles"]
        verts_y, triangles_y = result["Y"]["verts"], result["Y"]["triangles"]
        interpolated_verts_x = result["inter_verts"]
        fname_x = result["fname_x"][0]
        fname_y = result["fname_y"][0]

        if fname_x == fname_y:
            continue

        if structure_name not in fname_x:
            continue

        # load click data
        pat_fname_x = fname_x.replace(structure_name, '').replace('_', '')
        pat_fname_y = fname_y.replace(structure_name, '').replace('_', '')
        click_x = np.load(join(click_directory, click_name, pat_fname_x + ".npy"))
        click_y = np.load(join(click_directory, click_name, pat_fname_y + ".npy"))

        # identify the closest point on each mesh to the click
        lookup_tree_x = KDTree(verts_x)
        distances_x, indices_x = lookup_tree_x.query(click_x)
        closest_point_x = verts_x[indices_x]
        indices_on_y = assignment[indices_x]

        lookup_tree_y = KDTree(verts_y)
        distances_y, indices_y = lookup_tree_y.query(click_y)
        closest_point_y = verts_y[indices_y]
        indices_on_x = assignmentinv[indices_y]

        x_click_on_y = verts_y[indices_on_y]
        y_click_on_x = verts_x[indices_on_x]

        # calculate the error
        mesh_error_x.append(distances_x)
        mesh_error_y.append(distances_y)
        corr_error_x.append(np.linalg.norm(closest_point_y - x_click_on_y))
        corr_error_y.append(np.linalg.norm(closest_point_x - y_click_on_x))

        # calculate registration error
        final_verts_x = interpolated_verts_x[..., -1]
        closest_final_x = final_verts_x[indices_x]
        reg_error.append(np.linalg.norm(closest_point_y - closest_final_x))
        
    return np.array(corr_error_x), np.array(reg_error), np.array(mesh_error_x)

click_directory = ""
nrr_mesh_dir = ""
nrr_corrs_dir = ""
results_dir = ""
pat_fnames = sorted(getDirs(nrr_mesh_dir))
click_names = ["BS_bright_spot", "SC_c1_rear", "ParL_styloid_tip", "MandL_lingula"]
associated_oar_names = ["brainstem", "spinal_cord", "parotid_lt", "mandible"]

with open("landmark_fname_pairs.pkl", "rb") as f:
    landmark_fname_pairs = pickle.load(f)

pairs_passed = []
for click_name, structure_name in zip(click_names, associated_oar_names):
    click_error = np.array([])
    p_p = []
    for ref_pat_fname in tqdm(pat_fnames):
        for pat_fname in pat_fnames:
    
            if ref_pat_fname == pat_fname or not os.path.exists(join(nrr_mesh_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.ply")) or not os.path.exists(join(nrr_mesh_dir, ref_pat_fname, f"{ref_pat_fname}_{structure_name}.ply")):
                continue
            
            if (pat_fname, ref_pat_fname) not in landmark_fname_pairs:
                continue

            p_p.append((pat_fname, ref_pat_fname))

            # load meshes
            ref_mesh = o3d.io.read_triangle_mesh(join(nrr_mesh_dir, ref_pat_fname, f"{ref_pat_fname}_{structure_name}.ply"))
            nrr_mesh = o3d.io.read_triangle_mesh(join(nrr_mesh_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.ply"))

            # load correspondences
            assignment = np.load(join(nrr_corrs_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.npy"))

            # load click data
            click_y = np.load(join(click_directory, click_name, ref_pat_fname + ".npy"))
            click_x = np.load(join(click_directory, click_name, pat_fname + ".npy"))

            verts_y = np.asarray(ref_mesh.vertices)
            verts_x = np.asarray(nrr_mesh.vertices)

            # calculate results
            # identify the closest point on each mesh to the click
            lookup_tree_x = KDTree(verts_x)
            _, indices_x = lookup_tree_x.query(click_x)
            closest_point_x = verts_x[indices_x]

            indices_on_y = assignment[indices_x]
            x_click_on_y = verts_y[indices_on_y]

            lookup_tree_y = KDTree(verts_y)
            _, indices_y = lookup_tree_y.query(click_y)
            closest_point_y = verts_y[indices_y]

            corr_error = np.array([np.linalg.norm(closest_point_y - x_click_on_y)])
            click_error = np.concatenate([click_error, corr_error], axis=0)         

    print(f"Median error for {structure_name} is {np.median(click_error)}")
    # save results
    np.save(join(results_dir, f"{structure_name}_landmark_210.npy"), np.array(click_error))
    pairs_passed.append(p_p)

with open("fname_files/pairs_passed.pkl", "wb") as f:
    pickle.dump(pairs_passed, f)

