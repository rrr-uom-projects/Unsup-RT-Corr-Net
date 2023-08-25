import numpy as np
import open3d as o3d
import pygeodesic.geodesic as geodesic
import SimpleITK as sitk
import argparse as ap
from skimage.measure import marching_cubes
import scipy.ndimage as ndimage
from tqdm import tqdm

from edge_functions import split_long_edges, collapse_short_edges
import os
from os.path import join
from multiprocessing import Process
import warnings
warnings.filterwarnings("ignore")
from utils import getFiles, getDirs

def setup_argparse():
    parser = ap.ArgumentParser(prog="Sub program(?) for doing pre-processing in separate processes")
    parser.add_argument("--start_idx", type=int)
    parser.add_argument("--num_inds", type=int)
    parser.add_argument("--verbose", type=bool, default=False)
    args = parser.parse_args()
    return args

def connected_components(seg):
    # post-processing using scipy.ndimage.label to eliminate extraneous non-connected voxels
    labels, num_features = ndimage.label(input=seg, structure=np.ones((3,3,3)))
    sizes = ndimage.sum(seg, labels, range(num_features+1))
    seg[(labels!=np.argmax(sizes))] = 0
    return seg

def optimise_mesh(vertices, triangles, iterations=1000):
    for _ in range(iterations):
        vertices, triangles = split_long_edges(vertices, triangles, iterations=1)
        vertices, triangles = collapse_short_edges(vertices, triangles, iterations=1)
    return vertices, triangles

def f(pat_fname, lung_apex_cut, optimisation_iterations):
    global source_dir, output_meshes_dir, output_dists_dir, sdx, structure_name, outf_name
    # load mask
    mask = sitk.ReadImage(join(source_dir, pat_fname + ".nii"))
    spacing = np.array(mask.GetSpacing())[[2,1,0]]
    origin = np.array(mask.GetOrigin())[[2,1,0]]
    mask = sitk.GetArrayFromImage(mask)

    # select the correct structure
    mask = (mask == (sdx+1)).astype(float)

    # first apply connected components to remove extraneous non-connected voxels
    mask = connected_components(mask)

    # add check that the mask is not empty
    if np.sum(mask) == 0:
        print(f"Mask for {pat_fname} is empty")
        return
    
    # remove the structure below the apex of the lung contour (ensures that the spinal cord lengths are consistent)
    mask[:lung_apex_cut] = 0

    # add cc padding for the spinal cord - ensures mesh is closed at both ends - not sure if this is necessary or not
    if structure_name == "Spinal-Cord":
        # check if the mesh will be open at the caudal end
        min_level = np.min(np.argwhere(mask)[:,0])
        if min_level == 0:
            print(f"Mesh for {pat_fname} will be open at the caudal end - adding padding...")
            mask = np.pad(mask, ((1,1), (0,0), (0,0)), mode="constant", constant_values=0)

    # check again that the above operations haven't created an empty mask
    if np.sum(mask) == 0:
        raise RuntimeError(f"Mask for {pat_fname} is now empty - check the lung apex cut and spinal cord padding")

    # get the vertices and triangles
    vertices, faces, normals, _ = marching_cubes(volume=mask, level=0.49, spacing=spacing)

    # create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    # Initial smoothing
    num_triangles = 3000
    if "Submandibular" in structure_name:
        num_triangles = 2000
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=num_triangles)
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    mesh.remove_unreferenced_vertices()

    # optimise mesh with custom split and collapse methods
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if optimisation_iterations > 0:
        vertices, triangles = optimise_mesh(vertices, triangles, iterations=optimisation_iterations)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # get the vertices and triangles
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    n_vertices = vertices.shape[0]

    # compute geodesic distances between all pairs of vertices - do this beofre shifting to 0 CoM to keep geodesic distances in physical units (mm)
    geo_alg = geodesic.PyGeodesicAlgorithmExact(vertices, triangles)
    D = np.zeros((n_vertices, n_vertices), dtype=np.float32)
    for source_index in range(n_vertices):
        D[source_index], _ = geo_alg.geodesicDistances(np.array([source_index]))
    try:
        assert (D - D.T < 1e-4).all()
    except AssertionError: 
        print(f"Assertion error for {pat_fname} - max value: {np.abs(D - D.T).max()}", flush=True)
        # If nan -> It's likely the mesh is not a single connected component -> the orig contour is disconnected
        return

    # save the geodesic distances
    np.save(join(output_dists_dir, outf_name.replace('.ply', '.npy')), D)

    # put vertices back into the mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # save the mesh
    o3d.io.write_triangle_mesh(join(output_meshes_dir, outf_name), mesh)

def __main__():
    global source_dir, output_meshes_dir, output_dists_dir, sdx, structure_name, outf_name
    # get args
    args = setup_argparse()    
    # set directories
    root_source_dir = ""
    output_dir = ""
    os.makedirs(output_dir, exist_ok=True)

    structs = ["Brainstem", "Mandible", "Parotid-Lt", "Parotid-Rt", "Spinal-Cord", "Submandibular-Lt", "Submandibular-Rt"]
    # below will need adjusting according to your rigid prealignment results
    LUNG_APEX_CUTS = [38, 44, 42, 21, 42, 57, 32, 55, 65, 40, 46, 44, 59, 76, 92, 22, 31, 29, 14, 36, 311, 315, 39, 24, 42, 39, 50, 37, 44, 44, 50, 39, 32, 39]
    pat_fnames = sorted(getDirs(root_source_dir))

    fixed_pat_fnames = pat_fnames[args.start_idx:args.start_idx + args.num_inds]
    
    for fpdx, fixed_pat_fname in enumerate(tqdm(fixed_pat_fnames, disable=(not args.verbose))):
        if args.verbose:
            print(f"Starting {fixed_pat_fname}", flush=True)
        source_dir = join(root_source_dir, fixed_pat_fname)
        output_meshes_dir = join(output_dir, "meshes/", fixed_pat_fname)
        output_dists_dir = join(output_dir, "geodesic_distances/", fixed_pat_fname)
        os.makedirs(output_meshes_dir, exist_ok=True)
        os.makedirs(output_dists_dir, exist_ok=True)

        lung_apex_cut = LUNG_APEX_CUTS[fpdx+args.start_idx]
    
        for sdx, structure_name in enumerate(structs):
            #print(f"Starting {structure_name}", flush=True)
            optimisation_iterations = [1000, 500, 100, 0]
            for pat_fname in tqdm(pat_fnames):
                for opt_iter in optimisation_iterations:
                    outf_name = f"{pat_fname}_{structure_name.lower().replace('-', '_')}.ply"
                    process = Process(target=f, args=(pat_fname, lung_apex_cut, opt_iter))
                    process.start()
                    process.join()
                    if os.path.exists(join(output_meshes_dir, outf_name)) or opt_iter == 0:
                        break
                    #print(f"Mesh not created for {outf_name} - trying again with fewer optmisation iterations...")

            for pat_fname in pat_fnames:
                outf_name = f"{pat_fname}_{structure_name.lower().replace('-', '_')}.ply"
                if not os.path.exists(join(output_meshes_dir, outf_name)):
                    pass
                    #print(f"Mesh still not created for {outf_name} - likely empty mask or another error...")

        # Romeo Dunn
        print(f'Done {fixed_pat_fname}')

if __name__ == '__main__':
    __main__()

























