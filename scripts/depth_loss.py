import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import math
import tools
import cv2
import pandas as pd

def get_DAPI_dirs(dirpath):
    DAPIlist = {}
    for root, dirs, files in os.walk(dirpath):
        for d in dirs:
            if d == "DAPI":
                rootname = os.path.dirname(root)
                dapipath = os.path.join(root, d)
                DAPIlist[rootname] = get_tiff_data(dapipath)
    return DAPIlist


def get_tiff_data(dapipath):
    norm_tiffs = tools.get_files(dapipath)
    means = []
    for tiff in norm_tiffs:
        img = cv2.imread(tiff)
        imgBW = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBW = imgBW.astype(float)
        imgBW[imgBW == 0] = np.nan
        means.append(np.nanmean(imgBW))
    return means


def plot_depths(depths, means):
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

def main():
    norm_root_folder = sys.argv[1]
    dapi_dict = get_DAPI_dirs(norm_root_folder)
    df = pd.DataFrame(dapi_dict)
    print(df.head())


if __name__ == '__main__':
    main()
