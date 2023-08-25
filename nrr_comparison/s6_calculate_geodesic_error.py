## evaluate nrr correspondences

import numpy as np
import os
from os.path import join
from utils import getDirs
import open3d as o3d
from tqdm import tqdm

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


# load results from main_test script
structs = ["brainstem", "spinal_cord", "mandible"]
nrr_mesh_dir = ""
nrr_geodesic_dir = ""
nrr_corrs_dir = ""
results_dir = ""

pat_fnames = sorted(getDirs(nrr_corrs_dir))
for sdx, structure_name in enumerate(structs):
    print(f"Structure: {structure_name}")
    geo_error = np.array([])
    for ref_pat_fname in tqdm(pat_fnames):
        for pat_fname in pat_fnames:
    
            if ref_pat_fname == pat_fname or not os.path.exists(join(nrr_corrs_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.npy")):
                continue
            
            # load meshes
            ref_mesh = o3d.io.read_triangle_mesh(join(nrr_mesh_dir, ref_pat_fname, f"{ref_pat_fname}_{structure_name}.ply"))
            nrr_mesh = o3d.io.read_triangle_mesh(join(nrr_mesh_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.ply"))

            # load correspondences
            assignment = np.load(join(nrr_corrs_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.npy"))

            # load geodesic distances
            ref_geodesic_dists = np.load(join(nrr_geodesic_dir, ref_pat_fname, f"{ref_pat_fname}_{structure_name}.npy"))
            nrr_geodesic_dists = np.load(join(nrr_geodesic_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.npy"))

            # calculate mesh surface area
            ref_area = ref_mesh.get_surface_area()
            nrr_area = nrr_mesh.get_surface_area()

            ## evaluate the results
            # Correspondence
            geodesic_error = get_geodesic_error(geodesic_dists_x=nrr_geodesic_dists, geodesic_dists_y=ref_geodesic_dists, assignment=assignment, area_x=nrr_area, area_y=ref_area)
            geo_error = np.concatenate([geo_error, geodesic_error.reshape(-1)])

    # save results
    os.makedirs(results_dir, exist_ok=True)
    np.save(join(results_dir,  f"{structure_name}_geodesic_error.npy"), geo_error)

structs = ["parotid_lt", "parotid_rt"]
geo_error = np.array([])
for sdx, structure_name in enumerate(structs):
    print(f"Structure: {structure_name}")
    
    for ref_pat_fname in tqdm(pat_fnames):
        for pat_fname in pat_fnames:
    
            if ref_pat_fname == pat_fname or not os.path.exists(join(nrr_corrs_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.npy")):
                continue
            
            # load meshes
            ref_mesh = o3d.io.read_triangle_mesh(join(nrr_mesh_dir, ref_pat_fname, f"{ref_pat_fname}_{structure_name}.ply"))
            nrr_mesh = o3d.io.read_triangle_mesh(join(nrr_mesh_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.ply"))

            # load correspondences
            assignment = np.load(join(nrr_corrs_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.npy"))

            # load geodesic distances
            ref_geodesic_dists = np.load(join(nrr_geodesic_dir, ref_pat_fname, f"{ref_pat_fname}_{structure_name}.npy"))
            nrr_geodesic_dists = np.load(join(nrr_geodesic_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.npy"))

            # calculate mesh surface area
            ref_area = ref_mesh.get_surface_area()
            nrr_area = nrr_mesh.get_surface_area()

            ## evaluate the results
            # Correspondence
            geodesic_error = get_geodesic_error(geodesic_dists_x=nrr_geodesic_dists, geodesic_dists_y=ref_geodesic_dists, assignment=assignment, area_x=nrr_area, area_y=ref_area)
            geo_error = np.concatenate([geo_error, geodesic_error.reshape(-1)])

# save results
os.makedirs(results_dir, exist_ok=True)
np.save(join(results_dir,  f"parotid_geodesic_error.npy"), geo_error)

structs = ["submandibular_lt", "submandibular_rt"]
geo_error = np.array([])
for sdx, structure_name in enumerate(structs):
    print(f"Structure: {structure_name}")
    
    for ref_pat_fname in tqdm(pat_fnames):
        for pat_fname in pat_fnames:
    
            if ref_pat_fname == pat_fname or not os.path.exists(join(nrr_corrs_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.npy")):
                continue
            
            # load meshes
            ref_mesh = o3d.io.read_triangle_mesh(join(nrr_mesh_dir, ref_pat_fname, f"{ref_pat_fname}_{structure_name}.ply"))
            nrr_mesh = o3d.io.read_triangle_mesh(join(nrr_mesh_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.ply"))

            # load correspondences
            assignment = np.load(join(nrr_corrs_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.npy"))

            # load geodesic distances
            ref_geodesic_dists = np.load(join(nrr_geodesic_dir, ref_pat_fname, f"{ref_pat_fname}_{structure_name}.npy"))
            nrr_geodesic_dists = np.load(join(nrr_geodesic_dir, ref_pat_fname, f"{pat_fname}_{structure_name}.npy"))

            # calculate mesh surface area
            ref_area = ref_mesh.get_surface_area()
            nrr_area = nrr_mesh.get_surface_area()

            ## evaluate the results
            # Correspondence
            geodesic_error = get_geodesic_error(geodesic_dists_x=nrr_geodesic_dists, geodesic_dists_y=ref_geodesic_dists, assignment=assignment, area_x=nrr_area, area_y=ref_area)
            geo_error = np.concatenate([geo_error, geodesic_error.reshape(-1)])

# save results
os.makedirs(results_dir, exist_ok=True)
np.save(join(results_dir,  f"submandibular_geodesic_error.npy"), geo_error)

