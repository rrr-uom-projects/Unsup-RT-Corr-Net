import os
import numpy as np
import open3d as o3d
from os.path import join
from scipy.spatial import KDTree
from tqdm import tqdm
from utils import getDirs

source_dir = ""
output_dir = ""
pat_fnames = sorted(getDirs(source_dir))

structs = ["brainstem", "mandible", "parotid_lt", "parotid_rt", "spinal_cord", "submandibular_lt", "submandibular_rt"]

# load reference mesh
for ref_pat_fname in tqdm(pat_fnames):
    for struct in structs:
        if not os.path.exists(join(source_dir, ref_pat_fname, f"{ref_pat_fname}_{struct}.ply")):
           continue
        ref_mesh = o3d.io.read_triangle_mesh(join(source_dir, ref_pat_fname, f"{ref_pat_fname}_{struct}.ply"))
        ref_verts = np.asarray(ref_mesh.vertices)
        lookup_tree = KDTree(ref_verts)

        # load deformed mesh
        for def_pat_fname in pat_fnames:
            if ref_pat_fname == def_pat_fname or not os.path.exists(join(source_dir, ref_pat_fname, f"{def_pat_fname}_{struct}.ply")):
                continue
            def_mesh = o3d.io.read_triangle_mesh(join(source_dir, ref_pat_fname, f"{def_pat_fname}_{struct}.ply"))
            def_verts = np.asarray(def_mesh.vertices)

            # find nearest neighbour correspondences
            _, indices = lookup_tree.query(def_verts, k=1)

            # save correspondences
            os.makedirs(join(output_dir, ref_pat_fname), exist_ok=True)
            np.save(join(output_dir, ref_pat_fname, f"{def_pat_fname}_{struct}.npy"), indices)
