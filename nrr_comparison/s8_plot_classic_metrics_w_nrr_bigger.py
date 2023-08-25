import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib
matplotlib.rcParams.update({"font.size": 14})
from os.path import join

structs = ["brainstem", "spinal_cord", "mandible", "parotid", "submandibular"]
structs_names = ["Brainstem", "Spinal cord", "Mandible", "Parotid glands", "Submandibular glands"]
fig, axs = plt.subplots(len(structs), 3, figsize=(13,13))
colors = ['xkcd:blue', 'xkcd:orange', 'xkcd:green', 'xkcd:purple', 'xkcd:brown', 'xkcd:pink', 'xkcd:gray', 'xkcd:olive', 'xkcd:cyan']

results_dir = ""
nrr_results_dir = ""

for sdx, structure_name in enumerate(structs):
    # load results
    geo_error_baseline = np.load(join(results_dir, "baseline/", f"{structure_name}_geodesic_error.npy"))
    chamfer_dists_baseline = np.load(join(results_dir, "baseline/", f"{structure_name}_chamfer_dists.npy"))
    distort_baseline = np.load(join(results_dir, "baseline/", f"{structure_name}_distort.npy"))
    geo_error_w_imaging = np.load(join(results_dir, "w_imaging/", f"{structure_name}_geodesic_error.npy"))
    chamfer_dists_w_imaging = np.load(join(results_dir, "w_imaging/", f"{structure_name}_chamfer_dists.npy"))
    distort_w_imaging = np.load(join(results_dir, "w_imaging/", f"{structure_name}_distort.npy"))
    geo_error_imaging_loss = np.load(join(results_dir, "imaging_loss/", f"{structure_name}_geodesic_error.npy"))
    chamfer_dists_imaging_loss = np.load(join(results_dir, "imaging_loss/", f"{structure_name}_chamfer_dists.npy"))
    distort_imaging_loss = np.load(join(results_dir, "imaging_loss/", f"{structure_name}_distort.npy"))

    # nrr geodesic results
    geo_error_nrr = np.load(join(nrr_results_dir, f"{structure_name}_geodesic_error.npy"))
    chamfer_dists_nrr = np.load(join(nrr_results_dir, f"{structure_name}_chamfer_dists.npy"))

    # plot
    axs[sdx, 0].plot(np.sort(geo_error_baseline), np.linspace(0, 100, len(geo_error_baseline)), color=colors[0], linestyle='dashed', zorder=2)
    axs[sdx, 0].plot(np.sort(geo_error_w_imaging), np.linspace(0, 100, len(geo_error_w_imaging)), color=colors[1], zorder=0)
    axs[sdx, 0].plot(np.sort(geo_error_imaging_loss), np.linspace(0, 100, len(geo_error_imaging_loss)), color=colors[2], zorder=0)
    axs[sdx, 0].plot(np.sort(geo_error_nrr), np.linspace(0, 100, len(geo_error_nrr)), color=colors[3], zorder=0)

    axs[sdx, 1].plot(np.sort(chamfer_dists_baseline), np.linspace(0, 100, len(chamfer_dists_baseline)), color=colors[0], linestyle='dashed', zorder=2)
    axs[sdx, 1].plot(np.sort(chamfer_dists_w_imaging), np.linspace(0, 100, len(chamfer_dists_w_imaging)), color=colors[1], zorder=0)
    axs[sdx, 1].plot(np.sort(chamfer_dists_imaging_loss), np.linspace(0, 100, len(chamfer_dists_imaging_loss)), color=colors[2], zorder=0)
    axs[sdx, 1].plot(np.sort(chamfer_dists_nrr), np.linspace(0, 100, len(chamfer_dists_nrr)), color=colors[3], zorder=0)

    axs[sdx, 2].plot(np.sort(distort_baseline), np.linspace(0, 100, len(distort_baseline)), color=colors[0], linestyle='dashed', zorder=2)
    axs[sdx, 2].plot(np.sort(distort_w_imaging), np.linspace(0, 100, len(distort_w_imaging)), color=colors[1], zorder=0)
    axs[sdx, 2].plot(np.sort(distort_imaging_loss), np.linspace(0, 100, len(distort_imaging_loss)), color=colors[2], zorder=0) 

    axs[sdx, 0].set_ylim(0, 100)
    axs[sdx, 1].set_ylim(0, 100)
    axs[sdx, 2].set_ylim(0, 100)
    axs[sdx, 0].set_ylabel(structs_names[sdx], labelpad=-2)

    axs[sdx, 0].set_xlim(0, 0.4)
    axs[sdx, 1].set_xlim(0, 8)

    if sdx != 4:
        axs[sdx, 0].set_xticklabels([])
        axs[sdx, 1].set_xticklabels([])
        axs[sdx, 2].set_xticklabels([])


axs[0, 0].text(-0.05, 110, r"% of matches")
axs[0, 1].text(-1.25, 110, r"% of points")
axs[0, 2].text(-0.15, 110, r"% of triangles")

axs[sdx, 0].set_xlabel("Geodesic error")
axs[sdx, 1].set_xlabel("Chamfer distance (mm)")
axs[sdx, 2].set_xlabel("conformal distortion")

# John Legend
labels = ["Neuromorph", "+Imaging features", "+Imaging loss", "NRR"]
m_s = []
for obdx, (color,label) in enumerate(zip(colors, labels)):
    if obdx == 0:
        m_s.append(mlines.Line2D([],[], marker=None, color=color, linestyle='dashed', linewidth=2, label=label))
    else:
        m_s.append(mlines.Line2D([],[], marker=None, color=color, linestyle='-', linewidth=2, label=label))
last = m_s.pop()
m_s.insert(0, last)
axs[0, 2].legend(ncol=1, handles=m_s, fontsize=14, loc='lower right')

plt.tight_layout()
plt.subplots_adjust(hspace=0.1, wspace=0.13, left=0.05, right=0.995, bottom=0.045)

plt.savefig('results_with_nrr.png')