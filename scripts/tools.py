import os
import sys
import numpy as np
from scipy.stats import skew
from matplotlib import pyplot as plt

# Function: get_int_input()
# Description: Gets integer from the user between min and max provided. Shows
#              prompt to user before every input. Handles erroneous input.
# Pre-Conditions: String prompt, integer minimum and maximum given to function
# Post-Conditions: Returned integer between minimum and maximum
def get_int_input(prompt, min, max):
    while True:
        answer = min - 1
        try:
            answer = int(input(prompt))
        except ValueError:
            print(f"Attempted input was not an integer. Please try again")
        if not (answer <= max and answer >= min):
            print(f"Input was not between {min} and {max}. Please try again")
        else:
            return answer

def smart_check_int_param(param_value, param_name, min_val, max_val):
    try:
        param_value = int(param_value)
    except:
        raise ValueError(f"Parameter {param_name} must be an integer.")
    if not (param_value <= max_val and param_value >= min_val):
        raise ValueError(f"Parameter {param_name} must be larger than {min_val-1} and smaller than {max_val+1}.")
    return param_value


def smart_make_dir(dirpath):
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)


def norm_dirname(dirpath, dirtype, create = False):
    if dirpath == 0:
        return dirpath
    if not os.path.isdir(dirpath):
        if create:
            os.mkdir(dirpath)
        else:
            raise ValueError(f"Provided {dirtype} directory '{dirpath}' doesn't exist")
    if dirpath[-1] == os.path.sep:
        return dirpath[:-1]
    else:
        return dirpath


# Function: get_files()
# Description: Gets all files of '.tiff' format in a folder
# Pre-Conditions: Path-like string to valid directory provided
# Post-Conditions: Returns list of path-like objects to existing tiffs within dir.
def get_files(folder):
    tiffs = []
    root_name = ""
    for root, dirs, files in os.walk(folder):
        for f in files:
            this_name = '_'.join(f.split("_")[:-1])
            if root_name == "":
                root_name = this_name
            elif this_name != root_name:
                raise ValueError("Cannot confirm all images in folder are from the same stack. Ensure tiff filenames are all 'imgname_zX.tiff'")
            if f.split('.')[-1] == 'tif' or f.split('.')[-1] == 'tiff':
                tiffs.append(os.path.join(root, f))
    if not len(tiffs):
        raise ValueError("No tiffs found!")
    return tiffs


# Function: min_max_scale()
# Description: Performs min-max scaling on img object (pixel array)
# Pre-Conditions: Img object provided
# Post-Conditions: Return img object with pixels min-max scaled.
def min_max_scale(img):
    # std_max = np.nanmean(img) + 4 * np.nanstd(img)
    # img[img > std_max] = std_max
    plt.hist(img.flatten())
    plt.show()

    np.seterr(all = 'raise')

    minimum = img.min()
    maximum = img.max()
    # The formula for min-max scaling:
    img = (img - minimum) / (maximum - minimum) if maximum - minimum != 0 else (img - minimum) / 1
    img *= 255
    if isinstance(img, np.ma.MaskedArray):
        img = np.ma.getdata(img)
    return img


def get_outlier_boundaries(data):
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    med = np.median(data)
    return med - 1.5*iqr, med + 1.5*iqr
