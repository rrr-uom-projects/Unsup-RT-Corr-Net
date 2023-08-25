import os
import pickle
from os.path import join
from utils import getFiles, getDirs
from tqdm import tqdm

def get_landmark_fnames(sequence_fnames, results_folder):
    fname_pairs = []
    for sequence_fname in sequence_fnames:
        with open(join(results_folder, sequence_fname), "rb") as f:
            result = pickle.load(f)

        fname_x = result["fname_x"][0]
        fname_y = result["fname_y"][0]

        if fname_x == fname_y:
            continue

        if structure_name not in fname_x:
            continue

        # load click data
        pat_fname_x = fname_x.split("_")[-1]
        pat_fname_y = fname_y.split("_")[-1]
        
        fname_pairs.append((pat_fname_x, pat_fname_y))
    return fname_pairs

# load results from main_test script
click_names = ["BS_bright_spot", "SC_c1_rear", "ParL_styloid_tip", "MandL_lingula"]
associated_oar_names = ["brainstem", "spinal_cord", "parotid_lt", "mandible"]
models_dir = ""
results_dir = ""
all = []
all_unsorted = []
for click_name, structure_name in zip(click_names, associated_oar_names):

    fname_pairs = []
    for fold in range(1, 6):
        model_name = list(filter(lambda x: f"all_oars_fold{fold}" in x, getDirs(models_dir)))[0]
        model_path = join(models_dir, model_name, "out/")
        results_folder = join(model_path, getDirs(model_path)[0], "corrs/")
        sequence_fnames = sorted(getFiles(results_folder))

        fname_pairs.extend(get_landmark_fnames(sequence_fnames, results_folder))
    all_unsorted.append(fname_pairs)
    all.append(sorted(fname_pairs))
assert all[0] == all[1] == all[2] == all[3]

with open("fname_files/landmark_fname_pairs.pkl", "wb") as f:
    pickle.dump(all[0], f)

with open("fname_files/landmark_fname_pairs_unsorted.pkl", "wb") as f:
    pickle.dump(all_unsorted, f)
 
