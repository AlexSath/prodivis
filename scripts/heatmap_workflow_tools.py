from matplotlib import pyplot as plt
import matplotlib.colors
import matplotlib.colorbar as colorbar
import numpy as np
import math
from matplotlib.gridspec import GridSpec



# Function:
# Description:
# Pre-Conditions:
# Post-Conditions:
def process_zmin_zmax(zmin, zmax, tiffs):
    zmin_out, zmax_out = zmin, zmax
    if zmin == None:
        zmin_out = 0
    if zmax == None:
        zmax_out = len(tiffs)
    if zmin != None and zmax != None and zmin >= zmax - 2:
        raise ValueError(f"ZMin {zmin} must be 2 smaller than ZMax {zmax}")
    if zmin != None:
        if zmin >= 0:
            if type(zmin) == float:
                zmin_out = math.floor(zmin * len(tiffs))
        else:
            raise ValueError(f"ZMin {zmin} must be greater than or equal to 0!")
    if zmax != None:
        if zmax <= len(tiffs):
            if type(zmax) == float or (zmax == 1 and type(zmin) == float):
                zmax_out = math.ceil(zmax * len(tiffs))
        else:
            raise ValueError(f"ZMax {zmax} must be less than or equal to {len(tiffs)} (the stack's size)")
    return zmin_out, zmax_out


# Function:
# Description:
# Pre-Conditions:
# Post-Conditions:
def plot_dist(means, medians, mins, maxs, vars, title, savepath, save_figs):
    plt.plot(np.arange(0, len(means)), means, color = "#f03b20", linewidth = 3, label = "Mean")
    plt.plot(np.arange(0, len(medians)), medians, color = "#ef8a62", linewidth = 3, label = "Median")
    plt.plot(np.arange(0, len(mins)), mins, color = "#636363", linewidth = 3, label = "Min")
    plt.plot(np.arange(0, len(maxs)), maxs, color = "#bdbdbd", linewidth = 3, label = "Max")
    plt.plot(np.arange(0, len(vars)), vars, color = "#2ca25f", linewidth = 3, label = "Variance")
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    if ymin >= 0:
        ax.set_ylim((0, ymax))
    plt.title(title)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim((xmin, xmax - 2))
    plt.xlabel("Optical Slice Number")
    plt.ylabel("Pixel Value")
    plt.legend(bbox_to_anchor = (1.04,1), borderaxespad = 0)
    if save_figs:
        plt.savefig(savepath, format = "pdf")
        print("figure saved")
    plt.show()


def plot_histograms(histograms, title, savepath, bin_labels, save_figs):
    plt.imshow(histograms, cmap = 'hot', interpolation = 'nearest', origin = 'lower', aspect = 'auto', norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.xlabel("Histogram Bin (Range of Pixel Values)")
    plt.ylabel("Slice Number")
    plt.title(title)
    ax = plt.gca()
    ax.set_xticks(np.arange(1, len(bin_labels) + 1))
    plt.xticks(np.arange(len(bin_labels)), bin_labels, rotation=45)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim((xmin, xmax-.5))
    if save_figs:
        plt.savefig(savepath, format = "pdf", bbox_inches = 'tight')
        print("figure saved")

# Function:
# Description:
# Pre-Conditions:
# Post-Conditions:
def plot_MSI(means, title, savepath, save_figs):
    means = means / means.max() * 100
    plt.plot(np.arange(0, len(means)), means, color = "#636363", linewidth = 3, label = "Mean")
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    if ymin >= 0:
        ax.set_ylim((0, ymax))
    plt.title(title)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim((0, xmax - 2))
    plt.xlabel("Optical Slice Number")
    plt.ylabel("Pixel Intensity (% max)")
    plt.legend(loc="lower left")
    if save_figs:
        plt.savefig(savepath, format = "pdf")
        print("figure saved")
    plt.show()


# Function:
# Description:
# Pre-Conditions:
# Post-Conditions:
def plot_MSI_grouped(means_soi, means_ns, means_soi_ns, title, savepath, save_figs):
    means_soi = means_soi/means_soi.max() *100
    means_ns = means_ns/means_ns.max() *100
    means_soi_ns = means_soi_ns/means_soi_ns.max() *100
    plt.plot(np.arange(0, len(means_soi)), means_soi, color = "#cb181d", linewidth = 3, label = "Signal of Interest (SOI)")
    plt.plot(np.arange(0, len(means_ns)), means_ns, color = "#ef8a62", linewidth = 3, label = "Normalization Signal (NS)")
    plt.plot(np.arange(0, len(means_soi_ns)), means_soi_ns, color = "#999999", linewidth = 3, label = "Normalized SOI")
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    if ymin >= 0:
        ax.set_ylim((0, ymax))
    plt.title(title)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim((0, xmax - 2))
    plt.xlabel("Optical Slice Number")
    plt.ylabel("Pixel Intensity (% max)")
    plt.legend(loc="lower left")
    if save_figs:
        plt.savefig(savepath, format = "pdf")
        print("figure saved")
    plt.show()


# Function:
# Description:
# Pre-Conditions:
# Post-Conditions:
def plot_MSI_soi_ns(soi, ns, means_soi, means_ns, title, savepath, save_figs):
    means_soi = means_soi/means_soi.max() *100
    means_ns = means_ns/means_ns.max() *100
    plt.plot(np.arange(0, len(means_soi)), means_soi, color = "#cb181d", linewidth = 3, label = soi)
    plt.plot(np.arange(0, len(means_ns)), means_ns, color = "#ef8a62", linewidth = 3, label = ns)
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    if ymin >= 0:
        ax.set_ylim((0, ymax))
    plt.title(title)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim((0, xmax - 2))
    plt.xlabel("Optical Slice Number")
    plt.ylabel("Pixel Intensity (% max)")
    plt.legend(loc="lower left")
    if save_figs:
        plt.savefig(savepath, format = "pdf")
        print("figure saved")
    plt.show()


# Function:
# Description:
# Pre-Conditions:
# Post-Conditions:
def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)


# Function:
# Description:
# Pre-Conditions:
# Post-Conditions:
def plot_MIP(imgs, title_base, savepath, save_figs):
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(4, 3, figure=fig)
    ax1 = fig.add_subplot(gs[:2, :-2])
    ax2 = fig.add_subplot(gs[-1, :-2])
    ax3 = fig.add_subplot(gs[-2, :-2])
    ax1.imshow(imgs[0], cmap = 'magma')
    ax2.imshow(imgs[1], cmap = 'magma')
    ax3.imshow(imgs[2], cmap = 'magma')
    ax1.set_title(f"{title_base} Top View")
    ax2.set_title(f"{title_base} Front View")
    ax3.set_title(f"{title_base} Side View")
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    if save_figs:
        plt.savefig(savepath, format = "pdf",bbox_inches = 'tight', orientation = 'portrait')
        print("figure saved")
