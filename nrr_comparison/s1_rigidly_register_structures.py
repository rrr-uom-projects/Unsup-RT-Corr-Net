import SimpleITK as sitk
from utils import getDirs
from tqdm import tqdm
from os.path import join

def resample_image_with_Tx(referenceImage, Tx, iimg):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(referenceImage)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(-1024)
    resampler.SetTransform(Tx)
    oimg = resampler.Execute(iimg)
    return oimg

source_dir = ""
ref_ct_fpath = ""
corr_xfm_dir = ""
ms_xfm_dir = ""
out_dir = ""
fnames = sorted(getDirs(source_dir))
fnames.remove('TCGA-CV-A6JY')
# below will need adjusting according to your rigid prealignment results
RIGID_REG_STYLE = ['c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'm', 'c', 'c', 'c', 'c', 'm', 'c', 'c', 'c', 'c', 'm', 'c', 'c', 'c', 'c', 'm', 'c', 'c', 'c', 'm', 'c', 'c', 'c', 'c', 'c', 'c']
structs = ["Brainstem", "Mandible", "Parotid-Lt", "Parotid-Rt", "Spinal-Cord", "Submandibular-Lt", "Submandibular-Rt"]
for fname, reg_style in tqdm(zip(fnames, RIGID_REG_STYLE)):
    # read image
    ref_ct = sitk.ReadImage(ref_ct_fpath)
    
    # load the pre-computed transform
    if reg_style == "c":
        tfm = sitk.ReadTransform(join(corr_xfm_dir, {fname}.tfm))
    elif reg_style == "m":
        tfm = sitk.ReadTransform(join(ms_xfm_dir, {fname}.tfm)) 
    
    # loop through structs
    for sdx, structure_name in enumerate(structs):    
        mask = sitk.ReadImage(join(source_dir, fname, structure_name + ".nrrd"))
        
        # crop long scans
        if fname == "0522c0727a" or fname == "0522c0727b":
            mask = mask[:, :, 280:]

        # apply transform to masks
        mask_resampled = resample_image_with_Tx(ref_ct, tfm, mask)

        # save struct
        sitk.WriteImage(mask_resampled, join(out_dir, f'{fname}_{structure_name}.nii'))