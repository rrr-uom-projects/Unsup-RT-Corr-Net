## Script to evaluate different models

# import libraries
import numpy as np
import pickle
from os.path import join
from utils.utils import getFiles
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from tqdm import tqdm
import matplotlib.lines as mlines

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
        
    return (corr_error_x, corr_error_y, mesh_error_x, mesh_error_y, reg_error)


# load results from main_test script
click_names = ["BS_bright_spot", "SC_c1_rear", "ParL_styloid_tip", "MandL_lingula"]
associated_oar_names = ["brainstem", "spinal_cord", "parotid_lt", "mandible"]
for click_name, structure_name in zip(click_names, associated_oar_names):
    fig, axs = plt.subplots(1, 1, figsize=(8,8))

    # add runs here
    model_path = ""
    click_directory = ""
    results_folder = join(model_path, "corrs/")

    sequence_fnames = sorted(getFiles(results_folder))

    click_error = calculate_click_error(sequence_fnames, results_folder, structure_name, click_directory, click_name)

    y = np.linspace(0, 100, len(click_error[0]))
    colors = ['xkcd:neon green', "xkcd:neon blue", 'xkcd:orange']
    axs.plot(np.sort(click_error[0]), y, color=colors[0])
    axs.plot(np.sort(click_error[4]), y, color=colors[1])
    axs.plot(np.sort(click_error[2]), y, color=colors[2])

    axs.set_ylim(0, 100)
    axs.set_xlabel("Distance (mm)")
    axs.set_ylabel(r"% of comparisons")

    # John Legend
    obs_labels = ["Correspondence error", "Registration error", "Mesh error"]
    m_s = []
    for obdx, (color,label) in enumerate(zip(colors,obs_labels)):
        m_s.append(mlines.Line2D([],[], mfc=color, marker='s', linestyle='None', mew=1, mec='k', markersize=10, label=label))
    axs.legend(ncol=1, handles=m_s, fontsize="10", loc='lower right')

    plt.savefig(f'{structure_name}.png')
