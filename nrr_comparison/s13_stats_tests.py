import numpy as np
from os.path import join
from scipy.stats import iqr, wilcoxon
import pickle

def get_sig(p_val):
    if p_val < 0.000005:
        return "*****"
    if p_val < 0.00005:
        return "****"
    if p_val < 0.0005:
        return "***"
    if p_val < 0.005:
        return "**"
    if p_val < 0.05:
        return "*"
    return "ns"

click_names = ["Pineal gland", "C1 base", "Styloid process", "Mandible lingula"]
associated_oar_names = ["brainstem", "spinal_cord", "parotid_lt", "mandible"]

results_dir = ""
nrr_results_dir = ""

medians = np.zeros((len(click_names), 4))
iqrs = np.zeros((len(click_names), 4))

with open("fname_files/pairs_passed.pkl", "rb") as f:
    pairs_passed = pickle.load(f)

with open("fname_files/landmark_fname_pairs_unsorted.pkl", "rb") as f:
    pairs_unsorted = pickle.load(f)

all = []
for sdx, (click_name, structure_name) in enumerate(zip(click_names, associated_oar_names)):
    inds = []
    for pdx, pair in enumerate(pairs_unsorted[sdx]):
        if pair not in pairs_passed[sdx]:
            inds.append(pdx)
    all.append(inds)

for sdx, (click_name, structure_name) in enumerate(zip(click_names, associated_oar_names)):
    # load results
    landmark_baseline = np.load(join(results_dir, "baseline/", f"{structure_name}_landmark.npy")).reshape(210, 3)[:,0]
    landmark_w_imaging = np.load(join(results_dir, "w_imaging/", f"{structure_name}_landmark.npy")).reshape(210, 3)[:,0]
    landmark_imaging_loss = np.load(join(results_dir, "imaging_loss/", f"{structure_name}_landmark.npy")).reshape(210, 3)[:,0]
    landmark_nrr = np.load(join(nrr_results_dir, f"{structure_name}_landmark_210.npy"))

    # remove non-intersecting pairs
    landmark_baseline = np.delete(landmark_baseline, all[sdx])
    landmark_w_imaging = np.delete(landmark_w_imaging, all[sdx])
    landmark_imaging_loss = np.delete(landmark_imaging_loss, all[sdx])

    assert len(landmark_baseline) == len(landmark_w_imaging) == len(landmark_imaging_loss) == len(landmark_nrr)

    # save median and iqrs
    medians[sdx] = np.array([np.round(np.median(landmark_baseline), decimals=1), np.round(np.median(landmark_w_imaging), decimals=1), np.round(np.median(landmark_imaging_loss), decimals=1), np.round(np.median(landmark_nrr), decimals=1)])
    iqrs[sdx] = np.array([np.round(iqr(landmark_baseline), decimals=1), np.round(iqr(landmark_w_imaging), decimals=1), np.round(iqr(landmark_imaging_loss), decimals=1), np.round(iqr(landmark_nrr), decimals=1)])
    
    # perform wilcoxon signed rank test
    p_val = wilcoxon(landmark_baseline, landmark_nrr, alternative='less')[1]
    print(f"{click_name} - baseline vs nrr: {p_val} - {get_sig(p_val)}")
    p_val = wilcoxon(landmark_w_imaging, landmark_nrr, alternative='less')[1]
    print(f"{click_name} - w_imaging vs nrr: {p_val} - {get_sig(p_val)}")
    p_val = wilcoxon(landmark_imaging_loss, landmark_nrr, alternative='less')[1]
    print(f"{click_name} - imaging_loss vs nrr: {p_val} - {get_sig(p_val)}")

# print results better
print("\nMedian landmark distances (mm)\n")
for i, j in enumerate(["baseline    ", "w_imaging   ", "imaging_loss", "nrr         "]):
    print(" & ".join([j] + [f"{medians[sdx, i]} ({iqrs[sdx, i]})" for sdx in range(4)] ))