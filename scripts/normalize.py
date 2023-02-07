import os
import re
import cv2
import numpy as np
import tools
import sys
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import math
from builtins import len


# Function:
# Description:
# Pre-Conditions:
# Post-Conditions:
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
    log_10s = math.floor(math.log(len(tiff_list), 10)) + 14

    for tiff, norm in zip(np.sort(tiff_list), np.sort(norm_list)):
        print(f"Normalizing {tiff_dirname} using {norm_dirname} ({count}/{len(tiff_list)})...", end = '\n')
        if ign_mono:
            norm_tiffs.append(mean_normalize(tiff, norm, out_dir, threshold, outlier_stddevs, raw_norm, log_10s, norm_bool))
        else:
            norm_tiffs.append(mean_normalize(tiff, norm, out_dir, threshold, outlier_stddevs, raw_norm, log_10s))
        count += 1
    
        #if scale:
            #scale_tiffs(norm_tiffs)
    print('')

    return norm_tiffs


# Function:
# Description:
# Pre-Conditions:
# Post-Conditions:
def tiff_mean(normpath, thresh, stddevs, raw_norm, norm_bool):
    normBW = cv2.cvtColor(cv2.imread(normpath), cv2.COLOR_BGR2GRAY)
    normBW = normBW.astype(float)
    if type(norm_bool) != int:
        normBW[norm_bool == 0] = np.nan
    normBW[normBW == 0] = np.nan
    if not raw_norm:
        normBW[normBW < thresh] = np.nan
        if stddevs != -1:
            normBW[normBW > np.mean(normBW) + stddevs * np.std(normBW)] = np.nan
    return np.nanmean(normBW.flatten())


# Function:
# Description:
# Pre-Conditions:
# Post-Conditions:
def tiff_dist(normpath, lower, upper, norm_bool):
    normBW = cv2.cvtColor(cv2.imread(normpath), cv2.COLOR_BGR2GRAY)
    normBW = normBW.astype(float)
    if type(norm_bool) != int:
        normBW[norm_bool == 0] = np.nan
    normBW[normBW == 0] = np.nan
    normBW[normBW < lower] = np.nan
    normBW[normBW > upper] = np.nan
    return np.nanmean(normBW.flatten()), np.nanmedian(normBW.flatten()), np.nanmax(normBW.flatten()), np.nanmin(normBW.flatten()), np.nanvar(normBW.flatten())


# Function:
# Description:
# Pre-Conditions:
# Post-Conditions:
def tiff_stats_thresh(normpath, lower, upper, norm_bool):
    normBW = cv2.cvtColor(cv2.imread(normpath), cv2.COLOR_BGR2GRAY)
    normBW = normBW.astype(float)
    if type(norm_bool) != int:
        normBW[norm_bool == 0] = np.nan
    normBW[normBW == 0] = np.nan
    normBW[normBW < lower] = np.nan
    normBW[normBW > upper] = np.nan
    return np.nanmean(normBW.flatten())


def thresh(normpath, lower, upper, norm_bool):
    normBW = cv2.cvtColor(cv2.imread(normpath), cv2.COLOR_BGR2GRAY)
    normBW = normBW.astype(float)
    if type(norm_bool) != int:
        normBW[norm_bool == 0] = np.nan
    normBW[normBW == 0] = np.nan
    normBW[normBW < lower] = np.nan
    normBW[normBW > upper] = np.nan
    flattened = normBW[np.logical_not(np.isnan(normBW))].flatten()
    return flattened

def thresh_multiply(normpath, norm_bool):
    normBW = cv2.cvtColor(cv2.imread(normpath), cv2.COLOR_BGR2GRAY)
    normBW = normBW.astype(float)
    if type(norm_bool) != int:
        normBW[norm_bool == 0] = np.nan
    normBW[normBW == 0] = np.nan
    return np.array(normBW.flatten())


def analyze_images(directory, threshold):
    below_threshold = []
    for image in os.listdir(directory):
        path = os.path.join(directory, image)
        img = cv2.imread(path, 0)
        img = img.astype(float)
        img[img < threshold] = np.nan
        if np.isnan(img).all():
            below_threshold.append(image)
    return below_threshold


def images_used(img_list, norm_tiffs, stack_tiffs):
    img_list.sort()
    print("number of images below threshold = ",len(img_list))
    print("number of images in stack = ", len(norm_tiffs))
    if len(norm_tiffs) == len(stack_tiffs):
        if len(img_list) == len(norm_tiffs):
            print("all images are below threshold")
        if len(img_list) < len(norm_tiffs):
            print("% of images above threshold = ", (len(norm_tiffs)-len(img_list))/len(norm_tiffs) *100)
    else: print("number of images in NS and SOI stack are not equal")
    return img_list

# Function:
# Description:
# Pre-Conditions:
# Post-Conditions:
def mean_normalize(tiffpath, normpath, out_dir, thresh, stddevs, raw_norm, len_log, norm_bool = 0):
    tiffZ = int(''.join(re.findall("[0-9]+", tiffpath.split(os.path.sep)[-1]))[-1 * len_log:])
    normZ = int(''.join(re.findall("[0-9]+", normpath.split(os.path.sep)[-1]))[-1 * len_log:])
    #"tiff_ZXXX"
    if tiffZ != normZ:
        raise ValueError(f"Z-value for reference tiff ({tiffZ}) is not equal to Z-value for normalization tiff ({normZ})")

    tiffBW = cv2.cvtColor(cv2.imread(tiffpath), cv2.COLOR_BGR2GRAY)
    tiffBW[tiffBW < thresh] = 0
    tiffBW[tiffBW > stddevs] = 0
    tiffBW = (tiffBW / tiff_mean(normpath, thresh, stddevs, raw_norm, norm_bool).astype(np.float64)).astype(np.uint8)
    
    savepath = os.path.join(out_dir, f"{os.path.basename(os.path.dirname(tiffpath))}" + \
                                     f"_{'n_' if not raw_norm else 'rn_'}{'' if type(norm_bool) == int else 'im_'}" + \
                                     f"{os.path.dirname(normpath).split(os.path.sep)[-1]}_mean_{tiffZ}.tiff")
    cv2.imwrite(savepath, tiffBW)
    return savepath
        
def detect_sharp_change(data, threshold=None, direction='both'):
    diff = np.diff(data)
    if threshold is None:
        threshold = np.std(diff)
    if direction == 'increase':
        sharp_change = np.where(diff > threshold)[0]
    elif direction == 'decrease':
        sharp_change = np.where(diff < -threshold)[0]
    else:
        sharp_increase = np.where(diff > threshold)[0]
        sharp_decrease = np.where(diff < -threshold)[0]
        sharp_change = np.concatenate((sharp_increase, sharp_decrease))
    return sharp_change

