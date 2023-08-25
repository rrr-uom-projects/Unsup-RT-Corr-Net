import random
import SimpleITK as sitk
from tqdm import tqdm
import os
from os.path import join
from utils import getDirs

def f(fname, reference_fname, optimise=True):
    global source_dir, output_dir, reference_scan, reference_mask
    # load scan
    scan = sitk.ReadImage(join(source_dir, fname, "CT_IMAGE.nrrd"))
    if scan.GetPixelIDTypeAsString() != "64-bit float":
        scan = sitk.Cast(scan, sitk.sitkFloat64)

    # crop long scans
    if fname == "0522c0727a" or fname == "0522c0727b":
        scan = scan[:, :, 280:]

    # perform rigid registration to determine the transform to apply to the mesh
    initial_transform = sitk.CenteredTransformInitializer(reference_scan, scan, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsCorrelation()
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetInitialTransform(initial_transform)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)    # SPIE                                  # examples 3
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetMetricFixedMask(reference_mask)
    output_transform = registration_method.Execute(fixed=reference_scan, moving=scan)
    output_transform = sitk.Euler3DTransform(output_transform)

    # save the transform
    sitk.WriteTransform(output_transform, join(output_dir, f"{fname}.tfm"))
    return

def __main__():
    global source_dir, output_dir, reference_scan, reference_mask
    source_dir = ""
    output_dir = ""
    fnames = sorted(getDirs(source_dir))

    # set and load the reference CT scan
    random.seed(1210)
    reference_fname = fnames[random.randint(0, len(fnames)-1)]
    print(f"Reference patient for rigid registration: {reference_fname}", flush=True)
    output_dir = join(output_dir, "to_" + reference_fname + "_Correlation")
    os.makedirs(output_dir, exist_ok=True)
    reference_scan = sitk.ReadImage(join(source_dir, reference_fname, "CT_IMAGE.nrrd"))
    # create mask of scan   
    reference_mask = sitk.BinaryThreshold(reference_scan, lowerThreshold=-400, upperThreshold=10000)

    for fname in tqdm(fnames):
        f(fname, reference_fname)
    
    # report that all tasks are completed
    print('Romeo Dunn')

__main__()