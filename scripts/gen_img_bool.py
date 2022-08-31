import os
import sys
import tools as t
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def get_folders(rootdir, names, inverse):
    dirs_found = []
    for root, dirs, files in os.walk(rootdir):
        for d in dirs:
            if not inverse and (d in names or d == names):
                dirs_found.append(os.path.join(root, d))
            elif inverse and d not in names and d != names:
                firstfile = os.listdir(os.path.join(root, d))[0]
                if '.tif' in firstfile and 'norm' not in firstfile and 'projection' not in firstfile:
                    dirs_found.append(os.path.join(root, d))
    return dirs_found

def get_img_norm_idxs(norm_slice_paths, bool_cutoff):
    img1 = cv2.cvtColor(cv2.imread(norm_slice_paths[0]), cv2.COLOR_BGR2GRAY)
    slice_bool_cumulative = np.empty(img1.shape, dtype = np.uint8)
    for slice_path in norm_slice_paths:
        norm_slice = cv2.cvtColor(cv2.imread(slice_path), cv2.COLOR_BGR2GRAY)
        norm_slice[norm_slice > 0] = 1
        slice_bool_cumulative = slice_bool_cumulative + norm_slice
    final_cumulative = slice_bool_cumulative / len(norm_slice_paths)
    bool_cumulative = final_cumulative
    bool_cumulative[final_cumulative >= bool_cutoff] = 1
    bool_cumulative[final_cumulative < bool_cutoff] = 0
    return bool_cumulative

def main():
    keyword = sys.argv[2]
    dirs = get_folders(sys.argv[1], keyword, 0)
    outdir = sys.argv[3]
    t.smart_make_dir(outdir)
    for d in dirs:
        signal_parent = os.path.basename(os.path.dirname(d))
        signal_outdir = os.path.join(outdir, signal_parent)
        t.smart_make_dir(signal_outdir)
        tiffs = t.get_files(d)
        img_boundary = get_img_norm_idxs(tiffs, 0.85)
        plt.imshow(img_boundary)
        plt.savefig(os.path.join(signal_outdir, f"{keyword}_boundary.png"))
        plt.close()


if __name__ == '__main__':
    main()
