import numpy as np
import open3d as o3d
import pygeodesic.geodesic as geodesic
import SimpleITK as sitk
from skimage.measure import marching_cubes
import scipy.ndimage as ndimage

import os
from os.path import join
import shutil
from multiprocessing import Process
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

from utils import getFiles, getDirs, windowLevelNormalize
from edge_functions import split_long_edges, collapse_short_edges

def connected_components(seg):
    # post-processing using scipy.ndimage.label to eliminate extraneous non-connected voxels
    labels, num_features = ndimage.label(input=seg, structure=np.ones((3,3,3)))
    sizes = ndimage.sum(seg, labels, range(num_features+1))
    seg[(labels!=np.argmax(sizes))] = 0
    return seg

def optimise_mesh(vertices, triangles, iterations=1000):
    for _ in tqdm(range(iterations)):
        vertices, triangles = split_long_edges(vertices, triangles, iterations=1)
        vertices, triangles = collapse_short_edges(vertices, triangles, iterations=1)
    return vertices, triangles

def sample_ct_on_vertices(vertices, ct, spacing, patch_size):
    # sample the ct on the vertices
    # vertices: (N, 3) array of vertices
    # transform the vertices from world coordinates to voxel coordinates
    vox_vertices = vertices / spacing
    # get the bounding box of the vertices
    patch_size = np.array(patch_size)
    low_bounds = patch_size // 2
    high_bounds = (patch_size + 1) // 2 
    im_patches = np.zeros((len(vox_vertices),) + tuple(patch_size))
    # sample a 3D patch on the ct for every node in the mesh
    for vert_idx, vertex in enumerate(vox_vertices):
        # use clamping to check the patch is inside the ct
        vertex_unclamped = vertex
        vertex = np.clip(vertex_unclamped, a_min=low_bounds, a_max=np.array(ct.shape) - high_bounds)
        if (np.round(vertex_unclamped) != np.round(vertex)).any():
            raise RuntimeError("Vertex patch outside of the CT!")
        vertex = np.round(vertex).astype(int)
        crop_lo = vertex - low_bounds
        crop_hi = vertex + high_bounds
        im_patches[vert_idx] = ct[crop_lo[0]:crop_hi[0], crop_lo[1]:crop_hi[1], crop_lo[2]:crop_hi[2]]
    return im_patches


def f(fname, corr_or_ms, lung_apex_cut, optimisation_iterations):
    global source_dir, output_dir, structure_name, outf_name, corr_xfm_dir, ms_xfm_dir
    print(f"Started {fname}", flush=True)
    # load mask
    mask = sitk.ReadImage(join(source_dir, fname, "segmentations/", structure_name + ".nrrd"))
    spacing = np.array(mask.GetSpacing())[[2,1,0]]
    origin = np.array(mask.GetOrigin())[[2,1,0]]
    mask = sitk.GetArrayFromImage(mask)

    # first apply connected components to remove extraneous non-connected voxels
    mask = connected_components(mask)

    # add check that the mask is not empty
    if np.sum(mask) == 0:
        print(f"Mask for {fname} is empty")
        return

    # remove the structure below the apex of the lung contour (ensures that the spinal cord lengths are consistent)
    mask[:lung_apex_cut] = 0

    # add cc padding for the spinal cord - ensures mesh is closed at both ends - not sure if this is necessary or not
    if structure_name == "Spinal-Cord":
        # check if the mesh will be open at the caudal end
        min_level = np.min(np.argwhere(mask)[:,0])
        if min_level == 0:
            print(f"Mesh for {fname} will be open at the caudal end - adding padding...")
            mask = np.pad(mask, ((1,1), (0,0), (0,0)), mode="constant", constant_values=0)

    # check again that the above operations haven't created an empty mask
    if np.sum(mask) == 0:
        raise RuntimeError(f"Mask for {fname} is now empty - check the lung apex cut and spinal cord padding")

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

    # save the image patches ahead of time here
    ct = sitk.ReadImage(join(source_dir, fname, "CT_IMAGE.nrrd"))
    ct = sitk.GetArrayFromImage(ct)
    ct = windowLevelNormalize(image=ct, level=40, window=350)
    im_patches = sample_ct_on_vertices(vertices, ct, spacing, patch_size=(7,19,19))

    # perform rigid registration with pre-computed transforms
    # load the pre-computed transform
    if corr_or_ms == "c":
        tfm = sitk.ReadTransform(join(corr_xfm_dir, {fname}.tfm))
    elif corr_or_ms == "m":
        tfm = sitk.ReadTransform(join(ms_xfm_dir, {fname}.tfm))
    center = np.array(sitk.Euler3DTransform(tfm).GetCenter())[[2,1,0]]
    translation = np.array(sitk.Euler3DTransform(tfm).GetTranslation())[[2,1,0]]
    matrix = np.array(sitk.Euler3DTransform(tfm).GetMatrix()).reshape(3, 3)
    matrix = np.rot90(matrix, 2)
    # apply the transform
    vertices += origin
    vertices -= center
    vertices -= translation
    vertices = np.matmul(vertices, matrix)
    vertices += center

    # compute geodesic distances between all pairs of vertices - do this beofre shifting to 0 CoM to keep geodesic distances in physical units (mm)
    geo_alg = geodesic.PyGeodesicAlgorithmExact(vertices, triangles)
    D = np.zeros((n_vertices, n_vertices), dtype=np.float32)
    for source_index in range(n_vertices):
        D[source_index], _ = geo_alg.geodesicDistances(np.array([source_index]))
    try:
        assert (D - D.T < 1e-4).all()
    except AssertionError: 
        print(f"Assertion error for {fname} - max value: {np.abs(D - D.T).max()}", flush=True)
        # If nan -> It's likely the mesh is not a single connected component -> the orig contour is disconnected
        return

    # save the geodesic distances
    np.save(join(output_dir, "geodesic_distances/", f"{outf_name}_{fname}.npy"), D)

    # shift to 0 CoM
    if structure_name == "Spinal-Cord":
        # Shift vertices to origin CoM in ap and lr direction
        CoM = np.mean(vertices, axis=0)
        shift = np.array([vertices[:, 0].max(), CoM[1], CoM[2]])
        vertices = vertices - shift
    else:
        # Shift vertices to origin CoM
        shift = np.mean(vertices, axis=0)
        vertices = vertices - shift

    # put vertices back into the mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # save the mesh
    o3d.io.write_triangle_mesh(join(output_dir, "meshes/", f"{outf_name}_{fname}.ply"), mesh)

    # save the image patches
    np.save(join(output_dir, "im_patches/", f"{outf_name}_{fname}.npy"), im_patches)

    # save the shift
    np.save(join(output_dir, "shifts/", f"{outf_name}_{fname}.npy"), shift)

    # Romeo Dunn
    print(f"Finished {fname}", flush=True)

def __main__():
    global source_dir, output_dir, structure_name, outf_name, corr_xfm_dir, ms_xfm_dir
    # set directories
    source_dir = ""
    output_dir = ""
    corr_xfm_dir = ""
    ms_xfm_dir = ""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(join(output_dir, "meshes"), exist_ok=True)
    os.makedirs(join(output_dir, "im_patches"), exist_ok=True)
    os.makedirs(join(output_dir, "geodesic_distances"), exist_ok=True)
    os.makedirs(join(output_dir, "shifts"), exist_ok=True)

    structs = ["Brainstem", "Mandible", "Parotid-Lt", "Parotid-Rt", "Spinal-Cord", "Submandibular-Lt", "Submandibular-Rt"]
    for structure_name in tqdm(structs):
        print(f"Starting {structure_name}")
        outf_name = structure_name.lower().replace("-", "_")
        fnames = sorted(getDirs(source_dir))
        fnames.remove('TCGA-CV-A6JY')
        # below will need adjusting according to your rigid prealignment results
        RIGID_REG_STYLE = ['c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'm', 'c', 'c', 'c', 'c', 'm', 'c', 'c', 'c', 'c', 'm', 'c', 'c', 'c', 'c', 'm', 'c', 'c', 'c', 'm', 'c', 'c', 'c', 'c', 'c', 'c']
        LUNG_APEX_CUTS = [38, 44, 42, 21, 42, 57, 32, 55, 65, 40, 46, 44, 59, 76, 92, 22, 31, 29, 14, 36, 311, 315, 39, 24, 42, 39, 50, 37, 44, 44, 50, 39, 32, 39]

        optimisation_iterations = [1000, 500, 100, 0]
        for fname, reg_style, lung_apex_cut in zip(fnames, RIGID_REG_STYLE, LUNG_APEX_CUTS):
            for opt_iter in optimisation_iterations:
                process = Process(target=f, args=(fname, reg_style, lung_apex_cut, opt_iter))
                process.start()
                process.join()
                if os.path.exists(join(output_dir, "meshes/", f"{outf_name}_{fname}.ply")) or opt_iter == 0:
                    break
                print(f"Mesh not created for {fname} - trying again with fewer optmisation iterations...")

        for fname in fnames:
            if not os.path.exists(join(output_dir, "meshes/", f"{outf_name}_{fname}.ply")):
                print(f"Mesh still not created for {fname} - likely empty mask or another error...")

        # report that all tasks are completed
        print(f'Done {structure_name}')

__main__()