import os
import cv2
import numpy as np
import tools
import sys
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

def matrix_mean_normalizer(objective, normalization):
    z_means = np.mean(np.mean(normalization, axis = 2), axis = 1)
    print(z_means)
    sys.exit()

# Function: threshold_normalize()
# Description: Normalizes each tiff from the tiff_list by averaging the pixel
#              intensity (if the pixel intensity is above the integer threshold)
#              of the tiffs in the norm_list through proportional division.
# Pre-Conditions: List of path-like strings for tiff and norm, integer threshold
# Post-Conditions: New directory filled with normalized tiffs created. Returned
#                  list of path-like strings to generated normalized tiffs.
def mean_normalizer(tiff_list, norm_list, threshold, outlier_stddevs, raw_norm):
    tiff_dirname = os.path.dirname(tiff_list[0]).split(os.path.sep)[-1]
    norm_dirname = os.path.dirname(norm_list[0]).split(os.path.sep)[-1]

    dirname = f"{tiff_dirname}_{'n_' if not raw_norm else 'rn_'}{norm_dirname}" + \
              f"{'' if threshold == 0 else f'_t{threshold}'}{'' if outlier_stddevs == -1 else f'_{outlier_stddevs}std'}"
    out_dir = os.path.abspath(os.path.join(os.path.dirname(tiff_list[0]), '..', dirname))
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    else:
        use_temp = tools.get_int_input(f"Folder already found at {out_dir}. Should normalized TIFs be regenerated (1-y; 0-n)? ", 0, 1)
        if not use_temp:
            return tools.get_files(out_dir)

    print(f"\nCreating normalized tiffs in {out_dir}.")
    norm_tiffs = []
    count = 1
    for tiff, norm in zip(tiff_list, norm_list):
        print(f"Normalizing {tiff_dirname} using {norm_dirname} ({count}/{len(tiff_list)})...", end = '\n')
        norm_tiffs.append(mean_normalize(tiff, norm, out_dir, threshold, outlier_stddevs, raw_norm))
        count += 1
    print('')

    return norm_tiffs


def tiff_mean(normpath, thresh, stddevs, raw_norm):
    normBW = cv2.cvtColor(cv2.imread(normpath), cv2.COLOR_BGR2GRAY)
    normBW = normBW.flatten().astype(float)
    if not raw_norm:
        normBW[normBW < thresh] = np.nan
        if stddevs != -1:
            normBW[normBW > np.mean(normBW) + stddevs * np.std(normBW)] = np.nan
    return np.nanmean(normBW)


def mean_normalize(tiffpath, normpath, out_dir, thresh, stddevs, raw_norm):
    tiffZ = tiffpath.split('_')[-1].split('.')[0]
    normZ = normpath.split('_')[-1].split('.')[0]
    if tiffZ != normZ:
        raise ValueError(f"Z-value for reference tiff ({tiffZ}) is not equal to Z-value for normalization tiff ({normZ})")

    tiffBW = cv2.cvtColor(cv2.imread(tiffpath), cv2.COLOR_BGR2GRAY)
    flattened = tiffBW.flatten()
    tiffBW[tiffBW < thresh] = 0;
    if stddevs != -1:
        tiffBW[tiffBW > np.mean(flattened) + stddevs * np.std(flattened)] = np.median(flattened)

    tiffBW = (tiffBW / tiff_mean(normpath, thresh, stddevs, raw_norm).astype(np.float64)).astype(np.uint8) * 50

    savepath = os.path.join(out_dir, f"{'_'.join(tiffpath.split(os.path.sep)[-1].split('_')[:-1])}_norm_{os.path.dirname(normpath).split(os.path.sep)[-1]}_mean_{tiffZ}.tiff")
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
