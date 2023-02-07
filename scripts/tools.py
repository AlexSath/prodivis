import os
import sys
import numpy as np
import scipy.stats as scistat
from scipy.stats import percentileofscore
from matplotlib import pyplot as plt
from ipywidgets import widgets
from ipywidgets import interact
from IPython.display import display
from ipywidgets import IntProgress
from ipywidgets import FloatSlider
from IPython.core.display import HTML
import cv2

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


# Function:
# Description:
# Pre-Conditions:
# Post-Conditions:
def smart_check_int_param(param_value, param_name, min_val, max_val):
    try:
        param_value = int(param_value)
    except:
        raise ValueError(f"Parameter {param_name} must be an integer.")
    if not (param_value <= max_val and param_value >= min_val):
        raise ValueError(f"Parameter {param_name} must be larger than {min_val-1} and smaller than {max_val+1}.")
    return param_value


# Function:
# Description:
# Pre-Conditions:
# Post-Conditions:
def smart_make_dir(dirpath):
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)


# Function: 
# Description:
# Pre-Conditions:
# Post-Conditions:
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
                raise ValueError(f"Cannot confirm all images in {folder} are from the same stack. Ensure tiff filenames are all 'imgname_zX.tiff'")
            if f.split('.')[-1] == 'tif' or f.split('.')[-1] == 'tiff':
                tiffs.append(os.path.join(root, f))
    if not len(tiffs):
        raise ValueError("No tiffs found!")
    tiffs.sort()
    return tiffs


# Function: get_keyword_files()
# Description: Gets all files with provided keyword in a folder.
# Pre-Conditions: Path-like string to valid directory provided
# Post-Conditions: Returns list of path-like objects to existing tiffs within dir.
def get_keyword_files(folder, key):
    filepaths = []
    root_name = ""
    for root, dirs, files in os.walk(folder):
        for f in files:
            if key in f:
                filepaths.append(os.path.join(root, f))
    if not len(filepaths):
        raise ValueError(f"No files with keyword {key} found!")
    filepaths.sort()
    return filepaths


# Function: min_max_scale()
# Description: Performs min-max scaling on img object (pixel array)
# Pre-Conditions: Img object provided
# Post-Conditions: Return img object with pixels min-max scaled.
def min_max_scale(img):
    np.seterr(all = 'raise')

    minimum = img.min()
    maximum = img.max()
    # The formula for min-max scaling:
    img = (img - minimum) / (maximum - minimum) if maximum - minimum != 0 else (img - minimum) / 1
    img *= 255
    if isinstance(img, np.ma.MaskedArray):
        img = np.ma.getdata(img)
    return img


def stack_gui(tiff_list, norm_list):
    tiff_dirname = os.path.basename(os.path.dirname(tiff_list[0]))
    norm_dirname = os.path.basename(os.path.dirname(norm_list[0]))
    parent_dir = os.path.dirname(os.path.dirname(tiff_list[0]))
    display(HTML("<style>.widget-label { font-size: 12px; }</style>"))
    image_paths = tiff_list
    zstack = []
    for path in image_paths:
         img = cv2.imread(path)
         if img is not None:
             img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
             zstack.append(img)
         else:
             print(f'Error reading image: {path}')
    
    image_number_slider = widgets.IntSlider(min=0, max=len(image_paths), step=1, value=1, description='Slice #')
    find_brightest_button = widgets.Button(description="Find Brightest Image")
    image_number_slider.layout = widgets.Layout(width='1000px', height='30px', font_size='0px')
    find_brightest_button.layout = widgets.Layout(width='200px', height='30px', font_size='10px')
    save_button = widgets.Button(description="Save Images")
    save_button.layout = widgets.Layout(width='200px', height='30px', font_size='10px')
    min_val_slider = widgets.IntSlider(min=0, max=255, step=1, value=0, description='Min')
    max_val_slider = widgets.IntSlider(min=0, max=255, step=1, value=255, description='Max')
    min_val_slider.layout = widgets.Layout(width='500px', height='30px', font_size='10px')
    max_val_slider.layout = widgets.Layout(width='500px', height='30px', font_size='10px')
    progress_bar = IntProgress(value=0, min=0, max=len(image_paths), step=1, description='Loading:') 
    
    def getAlphaBeta(img, min_val, max_val):
        min_percent = percentileofscore(img.flatten(), min_val, kind = 'weak')
        max_percent = percentileofscore(img.flatten(), max_val, kind = 'weak')
        minimum, maximum = np.percentile(img.flatten(), [min_percent, max_percent])
        alpha = 255 / (maximum - minimum)
        beta = -minimum * alpha
        return alpha, beta

    def view_image(image_number, min_val, max_val):
        img = zstack[image_number]
        img = img.astype(np.uint8)
        min_val_slider.max = img.max()
        max_val_slider.max = img.max()
        # max_val_slider.value = img.max()
        alpha, beta = getAlphaBeta(img, min_val, max_val)
        img2 = cv2.convertScaleAbs(img, alpha = alpha, beta = beta)

        figure = plt.figure(figsize = (20, 10))
        # plt.figure(figsize=(20,10))
        ax1 = plt.subplot2grid((2, 3), (0, 0))
        ax2 = plt.subplot2grid((2, 3), (1, 0))
        ax3 = plt.subplot2grid((2, 3), (0, 1), rowspan = 2, colspan = 2)
        
        ax1.imshow(img, cmap='gray')
        ax1.axis('off')
        
        ax2.imshow(img2, cmap = 'gray')
        ax2.axis('off')
        
        ax3.hist(img.ravel(), 255, [0, 255], \
                 color = 'r', alpha = 0.5, label = 'Normalized')
        ax3.hist(img2.ravel(), 255, [0, 255], \
                 color = 'b', alpha = 0.5, label = 'Scaled')
        ax3.set_xlabel('Image Pixel Intensity')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()

    def find_brightest(b):
        max_value = 0
        max_idx = 0
        progress_bar.value = 0
        for idx, path in enumerate(image_paths):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
            img = img[img != 0]
            mean = np.mean(img)
            if mean > max_value:
                max_value = mean
                max_idx = idx
            else:
                progress_bar.value += 1
        image_number_slider.value = max_idx
        print(f'Brightest Image: {max_idx}')
        return max_idx

    find_brightest_button.on_click(find_brightest)
    
    out = interact(view_image, image_number=image_number_slider, min_val=min_val_slider, max_val=max_val_slider)
    out.widget.children += (find_brightest_button, save_button, progress_bar)  
    
    def save_images(min_val, max_val):
        progress_bar.value = 0
        new_dir = os.path.join(parent_dir, f"scaled_{tiff_dirname}_n{norm_dirname}")
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        alpha, beta = getAlphaBeta(zstack[find_brightest(1)], min_val, max_val)
        for i, path in enumerate(image_paths):
            if path is not None:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.convertScaleAbs(img, alpha = alpha, beta = beta)
                img = img.astype(np.uint8)
                new_path = os.path.join(new_dir, os.path.basename(path).split(".")[0] + ".tiff")
                cv2.imwrite(new_path, img)
            else:
                print(f'Error reading image: {path}')
            progress_bar.value += 1
        print(f'Images saved in {new_dir}')

    
    save_button.on_click(lambda b: save_images(min_val_slider.value, max_val_slider.value))




     


