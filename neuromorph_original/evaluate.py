## Script to evaluate different models

# import libraries
import numpy as np
import torch
import pickle
from os.path import join
from utils.utils import getFiles
from utils.distortion import compute_distortion
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from tqdm import tqdm
import matplotlib.lines as mlines

## functions to test correspondence and interpolation
# 1. Correlation
# Geodesic distance normalised by the square root area of the mesh
def get_geodesic_error(geodesic_dists_x, geodesic_dists_y, assignment, area_x, area_y, n_sample=10000):

    geodesic_dists_x_prime = geodesic_dists_y[assignment]
    geodesic_dists_x_prime = geodesic_dists_x_prime[:, assignment]
    error = np.abs(geodesic_dists_x - geodesic_dists_x_prime)

    # errors normalised by the square root area of the mesh
    geodesic_error = error / np.sqrt(area_y)
    # sample n_sample points
    geodesic_error = geodesic_error.reshape(-1)
    geodesic_error_sample = np.random.choice(geodesic_error, n_sample, replace=False)
    return geodesic_error_sample

# 2. Interpolation
# a) Conformal distortion metric
def distortion_metric(interpolated_verts_x, verts_x, triangles_x):
    final_verts_x = interpolated_verts_x[..., -1]
    verts_x = torch.tensor(verts_x, dtype=torch.float32)
    final_verts_x = torch.tensor(final_verts_x, dtype=torch.float32)
    triangles_x = torch.tensor(triangles_x, dtype=torch.int64)
    distortion = compute_distortion(verts_x, final_verts_x, triangles_x)
    return distortion.numpy()

# b) reconstruction error - Chamfer distance
def get_chamfer_distance(interpolated_verts_x, verts_y):
    verts_x = interpolated_verts_x[..., -1]
    lookup_tree = KDTree(verts_y)
    distances, _ = lookup_tree.query(verts_x)
    return distances

def calculate_metrics(sequence_fnames, results_folder, geodesic_dir):
    geo_error = []
    chamfer_dists = []
    distort = []
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

        if "brainstem" not in fname_x:
            continue

        geodesic_dists_x = np.load(join(geodesic_dir, f"{fname_x}.npy"))
        geodesic_dists_y = np.load(join(geodesic_dir, f"{fname_y}.npy"))

        # construct mesh for surface area
        mesh_x = o3d.geometry.TriangleMesh()
        mesh_x.vertices = o3d.utility.Vector3dVector(verts_x)
        mesh_x.triangles = o3d.utility.Vector3iVector(triangles_x)
        mesh_y = o3d.geometry.TriangleMesh()
        mesh_y.vertices = o3d.utility.Vector3dVector(verts_y)
        mesh_y.triangles = o3d.utility.Vector3iVector(triangles_y)
        area_x = mesh_x.get_surface_area()
        area_y = mesh_y.get_surface_area()

        ## evaluate the results
        # Correspondence
        geodesic_error = get_geodesic_error(geodesic_dists_x, geodesic_dists_y, assignment, area_x, area_y)
        geo_error.extend(geodesic_error.reshape(-1).tolist())

        # Interpolation
        chamfer_distances = get_chamfer_distance(interpolated_verts_x, verts_y)
        chamfer_dists.extend(chamfer_distances.tolist())    

        distortion = distortion_metric(interpolated_verts_x, verts_x, triangles_x)
        distortion = np.clip(distortion, 0, 1)
        distort.extend(distortion.tolist())
    return geo_error, chamfer_dists, distort


# load results from main_test script
fig, axs = plt.subplots(1, 3, figsize=(15,5))
geodesic_dir = ""
model_path = ""
results_folder = join(model_path, "corrs/")

sequence_fnames = sorted(getFiles(results_folder))

geo_error, chamfer_dists, distort = calculate_metrics(sequence_fnames, results_folder, geodesic_dir)

y = np.linspace(0, 100, len(geo_error))
axs[0].plot(np.sort(geo_error), y, 'g')
y = np.linspace(0, 100,  len(chamfer_dists))
axs[1].plot(np.sort(chamfer_dists), y, 'g')
y = np.linspace(0, 100,  len(distort))
axs[2].plot(np.sort(distort), y, 'g')

model_path = ""
results_folder = join(model_path, "corrs/")

sequence_fnames = sorted(getFiles(results_folder))

geo_error, chamfer_dists, distort = calculate_metrics(sequence_fnames, results_folder, geodesic_dir)

y = np.linspace(0, 100, len(geo_error))
axs[0].plot(np.sort(geo_error), y, 'r--')
y = np.linspace(0, 100,  len(chamfer_dists))
axs[1].plot(np.sort(chamfer_dists), y, 'r--')
y = np.linspace(0, 100,  len(distort))
axs[2].plot(np.sort(distort), y, 'r--')

model_path = ""
results_folder = join(model_path, "corrs/")

sequence_fnames = sorted(getFiles(results_folder))

geo_error, chamfer_dists, distort = calculate_metrics(sequence_fnames, results_folder, geodesic_dir)

y = np.linspace(0, 100, len(geo_error))
axs[0].plot(np.sort(geo_error), y, 'b-')
y = np.linspace(0, 100,  len(chamfer_dists))
axs[1].plot(np.sort(chamfer_dists), y, 'b-')
y = np.linspace(0, 100,  len(distort))
axs[2].plot(np.sort(distort), y, 'b-')

model_path = ""
results_folder = join(model_path, "corrs/")

sequence_fnames = sorted(getFiles(results_folder))

geo_error, chamfer_dists, distort = calculate_metrics(sequence_fnames, results_folder, geodesic_dir)

y = np.linspace(0, 100, len(geo_error))
axs[0].plot(np.sort(geo_error), y, 'm-')
y = np.linspace(0, 100,  len(chamfer_dists))
axs[1].plot(np.sort(chamfer_dists), y, 'm-')
y = np.linspace(0, 100,  len(distort))
axs[2].plot(np.sort(distort), y, 'm-')

model_path = ""
results_folder = join(model_path, "corrs/")

sequence_fnames = sorted(getFiles(results_folder))

geo_error, chamfer_dists, distort = calculate_metrics(sequence_fnames, results_folder, geodesic_dir)

y = np.linspace(0, 100, len(geo_error))
axs[0].plot(np.sort(geo_error), y, 'y-')
y = np.linspace(0, 100,  len(chamfer_dists))
axs[1].plot(np.sort(chamfer_dists), y, 'y-')
y = np.linspace(0, 100,  len(distort))
axs[2].plot(np.sort(distort), y, 'y-')

model_path = ""
results_folder = join(model_path, "corrs/")

sequence_fnames = sorted(getFiles(results_folder))

geo_error, chamfer_dists, distort = calculate_metrics(sequence_fnames, results_folder, geodesic_dir)

y = np.linspace(0, 100, len(geo_error))
axs[0].plot(np.sort(geo_error), y, color="xkcd:neon pink")
y = np.linspace(0, 100,  len(chamfer_dists))
axs[1].plot(np.sort(chamfer_dists), y, color="xkcd:neon pink")
y = np.linspace(0, 100,  len(distort))
axs[2].plot(np.sort(distort), y, color="xkcd:neon pink")

model_path = ""
results_folder = join(model_path, "corrs/")

sequence_fnames = sorted(getFiles(results_folder))

geo_error, chamfer_dists, distort = calculate_metrics(sequence_fnames, results_folder, geodesic_dir)

y = np.linspace(0, 100, len(geo_error))
axs[0].plot(np.sort(geo_error), y, color="xkcd:aquamarine")
y = np.linspace(0, 100,  len(chamfer_dists))
axs[1].plot(np.sort(chamfer_dists), y, color="xkcd:aquamarine")
y = np.linspace(0, 100,  len(distort))
axs[2].plot(np.sort(distort), y, color="xkcd:aquamarine")


axs[0].set_ylim(0, 100)
axs[1].set_ylim(0, 100)
axs[2].set_ylim(0, 100)

axs[0].set_xlabel("Geodesic error")
axs[0].set_ylabel(r"% of matches")
axs[1].set_xlabel("Chamfer distance (mm)")
axs[1].set_ylabel(r"% of points")
axs[2].set_xlabel("conformal distortion")
axs[2].set_ylabel(r"% of triangles")

# John Legend
obs_labels = ['baseline', r'$\lambda$ arap: 1000', r'$\lambda$ arap: 10', r'$\lambda$ arap: 0.1', 'hidden dim: 64', 'all_oars_lung_cut', 'all_oars']
colors = ['g', 'r', 'b', 'm', 'y', 'xkcd:neon pink', 'xkcd:aquamarine']
m_s = []
for obdx, (color,label) in enumerate(zip(colors,obs_labels)):
    m_s.append(mlines.Line2D([],[], mfc=color, marker='s', linestyle='None', mew=1, mec='k', markersize=10, label=label))
axs[2].legend(ncol=1, handles=m_s, fontsize="10", loc='lower right')

plt.savefig('metrics.png')
