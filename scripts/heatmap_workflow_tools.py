from matplotlib import pyplot as plt
import numpy as np
import math

def process_zmin_zmax(zmin, zmax, tiffs):
    zmin_out, zmax_out = zmin, zmax
    if zmin <= zmax - 2:
        raise ValueError(f"ZMin {zmin} must be 2 smaller than ZMax {zmax}")
    if zmin >= 0:
        if type(zmin) == float:
            zmin_out = math.floor(zmin * len(tiffs))
    else:
        raise ValueError(f"ZMin {zmin} must be greater than or equal to 0!")
    if zmax <= len(tiffs):
        if type(zmax) == float or (zmax == 1 and type(zmin) == float):
            zmax_out = math.ceil(zmax * len(tiffs))
    else:
        raise ValueError(f"ZMax {zmax} must be less than or equal to {len(tiffs)} (the stack's size)")
    return zmin_out, zmax_out

def plot_MSI(means, stds, depth, title, savepath):
    means, stds = means / means.max() * 100, stds / means.max() * 100
    plt.plot(np.arange(0, len(means)) * depth, means, color = "#8c510a", linewidth = 3)
    plt.fill_between(np.arange(0, len(means)) * depth, means + stds, means - stds, color = "#80cdc1")
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    if ymin >= 0:
        ax.set_ylim((0, ymax))
    plt.title(title)
    plt.xlabel("Slice Depth (um)")
    plt.ylabel("Intensity (% max)")
    plt.savefig(savepath, format = "png")
    
def plot_MIP(imgs, title_base, savepath):
    fig = plt.figure(figsize = (21, 7))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(imgs[0], cmap = 'magma')
    ax2.imshow(imgs[1], cmap = 'magma')
    ax3.imshow(imgs[2], cmap = 'magma')
    ax1.set_title(f"{title_base} MIP Z-View")
    ax2.set_title(f"{title_base} MIP X-View")
    ax3.set_title(f"{title_base} MIP Y-View")
    plt.tight_layout()
    plt.savefig(savepath, format = "png")