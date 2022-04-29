import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import math
import tools
import cv2

def main():
    norm_folder = sys.argv[1]
    norm_tiffs = tools.get_files(norm_folder)   
    means = []
    depths = []
    for idx, tiff in enumerate(norm_tiffs):
        img = cv2.imread(tiff)
        imgBW = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBW = imgBW.astype(float)
        imgBW[imgBW == 0] = np.nan
        means.append(np.nanmean(imgBW))
        depths.append(idx / len(norm_tiffs))

    fig = plt.figure(figsize = (int(sys.argv[4]), int(sys.argv[5])))
    ax = fig.add_subplot(111)
    label_format = '{}'

    ax.plot(depths, means, linewidth = 3)
    ax.set_ylim([0, round(max(means) + (np.std(means) / 5), 2)])

    ax.set_xlabel('% Depth Across Tumor', fontsize = 24)
    xticks = np.linspace(0, 10, 5) / 10
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize = 20)

    ax.set_ylabel(sys.argv[3], fontsize = 24)
    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels([label_format.format(y) for y in ax.get_yticks().tolist()], fontsize = 20)

    plt.savefig(sys.argv[2], format = 'png')

if __name__ == '__main__':
    main()
