import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import matplotlib.lines as mlines
import matplotlib
matplotlib.rcParams.update({"font.size": 16})
from scipy.stats import iqr
import pickle

click_names = ["Pineal gland", "C1 base", "Styloid process", "Mandible lingula"]
associated_oar_names = ["brainstem", "spinal_cord", "parotid_lt", "mandible"]
fig, axs = plt.subplots(ncols=1, figsize=(15,5.8))
colors = ['xkcd:blue', 'xkcd:orange', 'xkcd:green', 'xkcd:purple', 'xkcd:brown', 'xkcd:pink', 'xkcd:gray', 'xkcd:olive', 'xkcd:cyan']

results_dir = ""
nrr_results_dir = ""
medians = np.zeros((len(click_names), 5))
iqrs = np.zeros((len(click_names), 5))

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
    landmark_baseline = np.load(join(results_dir, "baseline/", f"{structure_name}_landmark.npy")).reshape(210, 3)
    landmark_w_imaging = np.load(join(results_dir, "w_imaging/", f"{structure_name}_landmark.npy")).reshape(210, 3)
    landmark_imaging_loss = np.load(join(results_dir, "imaging_loss/", f"{structure_name}_landmark.npy")).reshape(210, 3)
    landmark_nrr = np.load(join(nrr_results_dir, f"{structure_name}_landmark_210.npy"))

    # remove non-intersecting pairs
    landmark_baseline = np.delete(landmark_baseline, all[sdx], axis=0)
    landmark_w_imaging = np.delete(landmark_w_imaging, all[sdx], axis=0)
    landmark_imaging_loss = np.delete(landmark_imaging_loss, all[sdx], axis=0)

    assert len(landmark_baseline) == len(landmark_w_imaging) == len(landmark_imaging_loss) == len(landmark_nrr)

    # plot
    pos = np.arange(4) + (5 * sdx)
    pos = pos.astype(float)
    bplot = axs.boxplot([landmark_baseline[:, 0], landmark_w_imaging[:, 0], landmark_imaging_loss[:, 0], landmark_nrr], positions=pos, widths=0.8, patch_artist=True, zorder=0)
    for patch, line, color in zip(bplot['boxes'], bplot["medians"], colors):
        line.set_color("xkcd:white")
        line.set_linewidth(2.5)
        patch.set_facecolor(color)
    pos[0] -= 0.5
    pos[-1] += 0.5
    axs.plot(pos, np.repeat(np.median(landmark_baseline[:, 2]), 4), color='r', linestyle='--', linewidth=2.5, zorder=2)

    # save median and iqrs
    medians[sdx] = np.array([np.round(np.median(landmark_baseline[:,0]), decimals=1), np.round(np.median(landmark_w_imaging[:,0]), decimals=1), np.round(np.median(landmark_imaging_loss[:,0]), decimals=1), np.round(np.median(landmark_nrr), decimals=1), np.round(np.median(landmark_baseline[:,2]), decimals=1)])
    iqrs[sdx] = np.array([np.round(iqr(landmark_baseline[:,0]), decimals=1), np.round(iqr(landmark_w_imaging[:,0]), decimals=1), np.round(iqr(landmark_imaging_loss[:,0]), decimals=1), np.round(iqr(landmark_nrr), decimals=1), np.round(iqr(landmark_baseline[:,2]), decimals=1)])
    

axs.set_ylabel("Target landmark distance (mm)")
axs.set_xticks([1.5,6.5,11.5,16.5])
axs.set_xticklabels(click_names)

axs.set_ylim([0, 20.5])

# John Legend
labels = ["Baseline", "Imaging features", "Imaging loss", "NRR"]
m_s = []
for obdx, (color,label) in enumerate(zip(colors, labels)):
    m_s.append(mlines.Line2D([],[], mfc=color, marker='s', linestyle='None', mew=1, mec='k', markersize=15, label=label))
m_s.append(mlines.Line2D([],[], marker=None, linestyle='--', color='r', linewidth=2.5, label="Median distance from\nlandmark to organ"))
axs.legend(ncol=1, handles=m_s, loc='upper right')

plt.tight_layout()

plt.savefig('landmark_results_nrr.png')

# print results better
print("\nMedian landmark distances (mm)\n")
for i, j in enumerate(["baseline    ", "w_imaging   ", "imaging_loss", "nrr         ", "median_dist "]):
    print(" & ".join([j] + [f"{medians[sdx, i]} ({iqrs[sdx, i]})" for sdx in range(4)] ))