import os
import re
import cv2
import numpy as np
import tools
import sys
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import math

def matrix_mean_normalizer(objective, normalization):
    z_means = np.mean(np.mean(normalization, axis = 2), axis = 1)
    print(z_means)
    sys.exit()


def get_norm_bool_idxs(norm_slice_paths, bool_cutoff):
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

# Function: threshold_normalize()
# Description: Normalizes each tiff from the tiff_list by averaging the pixel
#              intensity (if the pixel intensity is above the integer threshold)
#              of the tiffs in the norm_list through proportional division.
# Pre-Conditions: List of path-like strings for tiff and norm, integer threshold
# Post-Conditions: New directory filled with normalized tiffs created. Returned
#                  list of path-like strings to generated normalized tiffs.
def mean_normalizer(tiff_list, norm_list, threshold, outlier_stddevs, raw_norm, ign_mono, ign_thresh = 0.7):
    tiff_dirname = os.path.basename(os.path.dirname(tiff_list[0]))
    norm_dirname = os.path.basename(os.path.dirname(norm_list[0]))

    dirname = f"{tiff_dirname}_{'n_' if not raw_norm else 'rn_'}" + \
              f"{'' if not ign_mono else 'im_'}{norm_dirname}" + \
              f"{'' if threshold == 0 else f'_t{threshold}'}" + \
              f"{'' if outlier_stddevs == -1 else f'_{outlier_stddevs}std'}"

    out_dir = os.path.abspath(os.path.join(os.path.dirname(tiff_list[0]), '..', dirname))
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    else:
        use_temp = tools.get_int_input(f"Folder already found at {out_dir}. Should normalized TIFs be regenerated (1-y; 0-n)? ", 0, 1)
        if not use_temp:
            return tools.get_files(out_dir)

    if ign_mono:
        norm_bool = get_norm_bool_idxs(norm_list, ign_thresh)

    print(f"\nCreating normalized tiffs in {out_dir}.")
    norm_tiffs = []
    count = 1
    log_10s = math.floor(math.log(len(tiff_list), 10)) + 1
    for tiff, norm in zip(np.sort(tiff_list), np.sort(norm_list)):
        print(f"Normalizing {tiff_dirname} using {norm_dirname} ({count}/{len(tiff_list)})...", end = '\n')
        if ign_mono:
            norm_tiffs.append(mean_normalize(tiff, norm, out_dir, threshold, outlier_stddevs, raw_norm, log_10s, norm_bool))
        else:
            norm_tiffs.append(mean_normalize(tiff, norm, out_dir, threshold, outlier_stddevs, raw_norm, log_10s))
        count += 1
    print('')

    return norm_tiffs

def tiff_mean_std(normpath, thresh, stddevs, raw_norm, norm_bool):
    normBW = cv2.cvtColor(cv2.imread(normpath), cv2.COLOR_BGR2GRAY)
    normBW = normBW.astype(float)
    if type(norm_bool) != int:
        normBW[norm_bool == 0] = np.nan
    normBW[normBW == 0] = np.nan
    if not raw_norm:
        normBW[normBW < thresh] = np.nan
        if stddevs != -1:
            normBW[normBW > np.mean(normBW) + stddevs * np.std(normBW)] = np.nan
    return np.nanmean(normBW.flatten()), np.nanstd(normBW.flatten())


def mean_normalize(tiffpath, normpath, out_dir, thresh, stddevs, raw_norm, len_log, norm_bool = 0):
    tiffZ = int(''.join(re.findall("[0-9]+", tiffpath.split(os.path.sep)[-1]))[-1 * len_log:])
    normZ = int(''.join(re.findall("[0-9]+", normpath.split(os.path.sep)[-1]))[-1 * len_log:])
    if tiffZ != normZ:
        raise ValueError(f"Z-value for reference tiff ({tiffZ}) is not equal to Z-value for normalization tiff ({normZ})")

    tiffBW = cv2.cvtColor(cv2.imread(tiffpath), cv2.COLOR_BGR2GRAY)
    flattened = tiffBW.flatten()
    tiffBW[tiffBW < thresh] = 0;
    if stddevs != -1:
        tiffBW[tiffBW > np.mean(flattened) + stddevs * np.std(flattened)] = np.median(flattened)

    tiffBW = (tiffBW / tiff_mean(normpath, thresh, stddevs, raw_norm, norm_bool).astype(np.float64)).astype(np.uint8) * 50

    savepath = os.path.join(out_dir, f"{os.path.basename(os.path.dirname(tiffpath))}" + \
                                     f"_{'n_' if not raw_norm else 'rn_'}{'' if type(norm_bool) == int else 'im_'}" + \
                                     f"{os.path.dirname(normpath).split(os.path.sep)[-1]}_mean_{tiffZ}.tiff")
    cv2.imwrite(savepath, tiffBW)
    return savepath

# Cell Normalizer Not Working Yet (more work needed with neural networks...)
def cell_normalizer(tiff_list, norm_list, phalloidin_list, prototxt, model, threshold, blur_matrix = (50, 50)):
    tiff_dirname = os.path.dirname(tiff_list[0]).split(os.path.sep)[-1]
    norm_dirname = os.path.dirname(norm_list[0]).split(os.path.sep)[-1]
    phalloidin_dirname = os.path.dirname(phalloidin_list[0]).split(os.path.sep)[-1]

    out_dir = os.path.abspath(os.path.join(os.path.dirname(tiff_list[0]), '..', f"{tiff_dirname}_norm_{norm_dirname}_cellSpecific_t{threshold}"))
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    else:
        use_temp = tools.get_int_input(f"Folder already found at {out_dir}. Should normalized TIFs be regenerated (1-y; 0-n)? ", 0, 1)
        if not use_temp:
            return tools.get_files(out_dir)

    lp_filtered_dir = os.path.join(out_dir, 'lp_filtered')
    tools.smart_make_dir(lp_filtered_dir)
    lp_filtered_list = []
    for phall in phalloidin_list:
        lp_filtered_list.append(lp_filter(phall, lp_filtered_dir, blur_matrix))

    cnn_canny_dir = os.path.join(out_dir, 'cnn_canny')
    tools.smart_make_dir(cnn_canny_dir)
    cnn_canny_list = []
    for lp_filtered in lp_filtered_list:
        lpfZ = lp_filtered.split('_')[-1].split('.')[0]
        savepath = os.path.join(cnn_canny_dir, f"{'_'.join(lp_filtered.split(os.path.sep)[-1].split('_')[:-1])}_CNN_canny_{lpfZ}.tiff")
        res = subprocess.run(['python', 'canny_cnn.py', '-i', lp_filtered, '-o', savepath, '-p', prototxt, '-m', model], shell = True, check = True)


def lp_filter(phallpath, out_dir, matrix):
    phallZ = phallpath.split('_')[-1].split('.')[0]
    phall = cv2.imread(phallpath)
    phall_lp = cv2.blur(phall, matrix)
    lp_filtered = cv2.subtract(phall, phall_lp)
    savepath = os.path.join(out_dir, f"{'_'.join(phallpath.split(os.path.sep)[-1].split('_')[:-1])}_lp_filtered_{phallZ}.tiff")
    cv2.imwrite(savepath, lp_filtered, bbox_inches = 'tight')
    return savepath


def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    prototxt = sys.argv[3]
    model = sys.argv[4]

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    tiffs = tools.get_files(input_dir)
    for tiff in tiffs:
        base = tiffs[0].split(os.path.sep)[-1].split('.')[0]
        output = os.path.join(output_dir, f'{base}_CNN-Canny.tiff')
        print(f"Running CNN Canny on {base}; outputting to {output}")
        result = subprocess.run(['python', 'canny_cnn.py', '-i', tiff, '-o', output, '-p', prototxt, '-m', model], shell = True, check = True)

if __name__ == '__main__':
    main()
