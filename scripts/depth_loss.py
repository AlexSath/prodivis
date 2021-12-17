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

    plt.plot(depths, means)
    plt.ylim([0, round(max(means) + (np.std(means) / 5), 2)])
    plt.xlabel('% Depth Across Tumor')
    plt.ylabel('Mean DAPI Pixel Intensity')
    plt.savefig(sys.argv[2], format = 'png')

if __name__ == '__main__':
    main()
