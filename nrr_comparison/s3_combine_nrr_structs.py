import numpy as np
import SimpleITK as sitk
import os
from os.path import join
from utils import getFiles
from tqdm import tqdm

rigid_dir = ""
nrr_dir = ""
fnames = sorted(getFiles(rigid_dir))
structs = ["Brainstem", "Mandible", "Parotid-Lt", "Parotid-Rt", "Spinal-Cord", "Submandibular-Lt", "Submandibular-Rt"]
out_dir = ""

for fixed_fname in tqdm(fnames):
    fixed_struct = sitk.ReadImage(join(rigid_dir, fixed_fname))
    fixed_struct_npy = sitk.GetArrayFromImage(fixed_struct)
    os.makedirs(join(out_dir, fixed_fname.replace('.nii', '')), exist_ok=True)
    for moving_fname in fnames:
        all_structs = np.zeros_like(fixed_struct_npy)
        for sdx, struct in enumerate(structs):
            moving_struct = sitk.ReadImage(join(nrr_dir, fixed_fname.replace('.nii', ''), moving_fname.replace('.nii', f'_{struct}.nii')))
            moving_struct = sitk.GetArrayFromImage(moving_struct)
            moving_struct = moving_struct > 0
            all_structs += moving_struct.astype(np.uint8) * (sdx + 1)
            all_structs = np.clip(all_structs, 0, sdx + 1)
        all_structs = sitk.GetImageFromArray(all_structs)
        all_structs.CopyInformation(fixed_struct)
        sitk.WriteImage(all_structs, join(out_dir, fixed_fname.replace('.nii', ''), moving_fname))