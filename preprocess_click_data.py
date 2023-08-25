import SimpleITK as sitk
import numpy as np
import os
from os.path import join
from scipy.ndimage import center_of_mass
from neuromorph_w_imaging.utils.utils import getFiles
from tqdm import tqdm

data_source = ""
clicks_source_dir = ""
clicks_target_dir = ""
shifts_source_dir = ""
registered_niftis_source_dir = ""

click_names = ["BS_bright_spot", "SC_c1_rear", "ParL_styloid_tip", "MandL_lingula"]
associated_oar_names = ["brainstem", "spinal_cord", "parotid_lt", "mandible"]

for click_name in click_names:
    os.makedirs(join(clicks_target_dir, click_name), exist_ok=True)

pat_fnames = sorted(getFiles(clicks_source_dir))

for pat_fname in tqdm(pat_fnames):
    landmarks = sitk.ReadImage(join(clicks_source_dir, pat_fname))
    ct = sitk.ReadImage(join(registered_niftis_source_dir, pat_fname.replace(".npy", "nii")))
    origin = np.array(ct.GetOrigin())[[2,1,0]]
    spacing = np.array(ct.GetSpacing())[[2,1,0]]
    landmarks = sitk.GetArrayFromImage(landmarks)
    for landmark_idx, (click_name, oar_name) in enumerate(zip(click_names, associated_oar_names)):
        click = landmarks == landmark_idx+1
        if click_name == "ParL_styloid_tip":
            click_cc = int(np.max(np.argwhere(click)[:,0]))
            click = np.array((click_cc,) + center_of_mass(click[click_cc]))
        else:
            click = np.array(center_of_mass(click))
        # load shift to zero CoM to bring click to same reference as the meshes
        click += origin / spacing
        CoM_shift = np.load(join(shifts_source_dir, oar_name + "_" + pat_fname.replace('.nii','.npy')))
        click -= CoM_shift / spacing
        np.save(join(clicks_target_dir, click_name, pat_fname.replace('.nii','.npy')), click)
