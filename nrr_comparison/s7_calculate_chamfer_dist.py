## evaluate nrr correspondences

import numpy as np
import os
from os.path import join
from utils import getDirs
import open3d as o3d
from scipy.spatial import KDTree
from tqdm import tqdm

def get_chamfer_distance(verts_x, verts_y):
    lookup_tree = KDTree(verts_y)
    distances_to_y, _ = lookup_tree.query(verts_x)
    lookup_tree = KDTree(verts_x)
    distances_to_x, _ = lookup_tree.query(verts_y)
    distances = np.concatenate([distances_to_x, distances_to_y])
    return distances

##########################################################################################

nrr_mesh_dir = ""
results_dir = ""
structs = ["brainstem", "spinal_cord", "mandible"]
pat_fnames = sorted(getDirs(nrr_mesh_dir))
for sdx, structure_name in enumerate(structs):
    print(f"Structure: {structure_name}")
    chamfer_dists = np.array([])
    for ref_pat_fname in tqdm(pat_fnames):
        for pat_fname in pat_fnames:
    
            if ref_pat_fname == pat_fname or not os.path.exists(join(nrr_mesh_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.ply")) or not os.path.exists(join(nrr_mesh_dir, ref_pat_fname, f"{ref_pat_fname}_{structure_name}.ply")):
                continue
            
            # load meshes
            ref_mesh = o3d.io.read_triangle_mesh(join(nrr_mesh_dir, ref_pat_fname, f"{ref_pat_fname}_{structure_name}.ply"))
            nrr_mesh = o3d.io.read_triangle_mesh(join(nrr_mesh_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.ply"))

            # Calculate chamfer distance
            chamfer_dist = get_chamfer_distance(np.asarray(nrr_mesh.vertices), np.asarray(ref_mesh.vertices))
            chamfer_dists = np.concatenate([chamfer_dists, chamfer_dist])

    # save results
    os.makedirs(results_dir, exist_ok=True)
    np.save(join(results_dir,  f"{structure_name}_chamfer_dists.npy"), chamfer_dists)

structs = ["parotid_lt", "parotid_rt"]
chamfer_dists = np.array([])
for sdx, structure_name in enumerate(structs):
    print(f"Structure: {structure_name}")
    
    for ref_pat_fname in tqdm(pat_fnames):
        for pat_fname in pat_fnames:
            
            if ref_pat_fname == pat_fname or not os.path.exists(join(nrr_mesh_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.ply")) or not os.path.exists(join(nrr_mesh_dir, ref_pat_fname, f"{ref_pat_fname}_{structure_name}.ply")):
                continue
            
            # load meshes
            ref_mesh = o3d.io.read_triangle_mesh(join(nrr_mesh_dir, ref_pat_fname, f"{ref_pat_fname}_{structure_name}.ply"))
            nrr_mesh = o3d.io.read_triangle_mesh(join(nrr_mesh_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.ply"))

            # Calculate chamfer distance
            chamfer_dist = get_chamfer_distance(np.asarray(nrr_mesh.vertices), np.asarray(ref_mesh.vertices))
            chamfer_dists = np.concatenate([chamfer_dists, chamfer_dist])

# save results
os.makedirs(results_dir, exist_ok=True)
np.save(join(results_dir,  f"parotid_chamfer_dists.npy"), chamfer_dists)

structs = ["submandibular_lt", "submandibular_rt"]
chamfer_dists = np.array([])
for sdx, structure_name in enumerate(structs):
    print(f"Structure: {structure_name}")
    
    for ref_pat_fname in tqdm(pat_fnames):
        for pat_fname in pat_fnames:
    
            if ref_pat_fname == pat_fname or not os.path.exists(join(nrr_mesh_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.ply")) or not os.path.exists(join(nrr_mesh_dir, ref_pat_fname, f"{ref_pat_fname}_{structure_name}.ply")):
                continue
            
            # load meshes
            ref_mesh = o3d.io.read_triangle_mesh(join(nrr_mesh_dir, ref_pat_fname, f"{ref_pat_fname}_{structure_name}.ply"))
            nrr_mesh = o3d.io.read_triangle_mesh(join(nrr_mesh_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.ply"))

            # Calculate chamfer distance
            chamfer_dist = get_chamfer_distance(np.asarray(nrr_mesh.vertices), np.asarray(ref_mesh.vertices))
            chamfer_dists = np.concatenate([chamfer_dists, chamfer_dist])

# save results
os.makedirs(results_dir, exist_ok=True)
np.save(join(results_dir,  f"submandibular_chamfer_dists.npy"), chamfer_dists)
