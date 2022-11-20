from matplotlib import pyplot as plt
import numpy as np

def plot_MSI(means, stds, depth, title, savepath):
    means, stds = means / means.max() * 100, stds / means.max() * 100
    plt.plot(np.arange(0, len(means)) * depth, means, color = "#8c510a", linewidth = 3)
    plt.fill_between(np.arange(0, len(means)) * depth, means + stds, means - stds, color = "#80cdc1")
    ax = plt.gca()
    ymax = ax.get_ylim()[1]
    ax.set_ylim((0, ymax))
    # ax.set_xlim(0, np.max(len(means) * depth))
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