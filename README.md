# ProDiVis
A Z-stack validation suite written in the python programming language, designed to be user-friendly for those with little knowledge of image analysis or programming. ProDiVis takes Z-stack outputs acquired from bioformat files (for example: CARL ZEISS .czi or Leica .lif) and generates a heatmap of differentially localized protein(s) normalized to a user-selected fluorescent housekeeping signal. Heatmaps can be generated from any fluorescent Z-tack and serve as an unbiased visualization tool. In addition to producing heatmaps, proDiVis outputs a normalized Z-stack that can be readily used with concurrent microscopy images. Built for data acquired in 'z-stack' form, where different optical sections are imaged individually.

## Inherent Problems with Confocal Deep Imaging
A common problem with laser scanning confocal microscopy is the intensity of excitation light and emission signal from fluorophores decrease as they travel through 3D specimens, resulting in weaker signal reaching the detector. To facilitate image interpretation, we developed proDiVis: a visualization algorithm involving focal-plane-specific signal normalization. However, there are limitations to optical sectioning while imaging thick specimens, namely fluorescence intensity loss as a function of imaging depth. The most widely used fluorophores have emission light in the visible spectrum that can only penetrate a limited distance through biological material. This is a known contributor to the fundamental depth limit, caused by several physical properties such as light scattering and absorption. To address the above issues, we developed a computational method to proportionally compare pixel values across the depth of 3D specimens, accounting for the decrease in fluorescence intensity we and others have previously observed. While there are many existing programs or software available for image analysis, to the best of our knowledge, none of them account for loss of signal by normalizing against a housekeeping signal, and many of them are expensive and/or require a high level of technical expertise to use.

## Dependencies
opencv, matplotlib, scipy, pandas, numpy, jupyter

`Data_Pipeline.ipynb` contains all the necessary code to run proDiVis

# Detailed Installation Instructions for Novice Users
We reccomend using python 3.11.3 (latest release as of April 5, 2013) which can be found
[here](https://www.python.org/downloads/)

We run proDiVis and subsequently python through [Anaconda](https://www.anaconda.com/), which is a software management tool designed to simplify the process of working with python and many python packages required for scientific computing. Using either method of installation works.

Navigate to the directory (folder) where you wish to install proDiVis

Using the command line: `git clone  https://github.com/AlexSath/prodivis.git`

You should have downloaded all the files needed to run proDiVis inside this folder. Before running proDiVis, you must create a virtual environment and install the required packages. 

## Using pip to create an environment
#### create a virtual environment inside the directory that proDiVis is located in, replace 'env_name' with the desired name of your virtual environment.

`python -m venv env_name`

#### Activate the environment

windows: `.\env_name/Scripts\activate`

macOS: `source env_name/bin/activate` 

#### install required packages

`pip install opencv-python matplotlib scipy pandas numpy jupyter`

## Using conda to create an environment
#### navigate to the directory (folder) where you wish to install proDiVis

`conda create --name env_name python=3.11.3`

#### activate the environment

`conda activate env_name`

#### install required packages

`conda install -c conda-forge opencv matplotlib scipy pandas numpy jupyter`

# Operation Instructions
#### Assuming the images needed are already saved in a bioformat file
#### 1. Export the images in .tiff format into a directory of your choosing. Ensure all filenames are `imgname_ZX.tiff` where `imgname` is the name of the image and `X` is the Z-stack number and focal plane number respectively. For example, `imgname_Z01.tiff` is the first focal plane of the Z-stack. It will be easier if the images are saved in a folder within the proDiVis directory.

#### 2. Open `Data_Pipeline.ipynb`

#### 3. Run the first cell to ensure that all necessary libraries are properly imported

#### 4. In the second cell, in quotes:
insert the path to your signal of interest (SOI) that will be assigned to the `stack_dir` variable
insert the path to your normalization signal (NS) that will be assigned to the `norm_dir` variable
insert the name of your soi and ns and assign them to the `soi` and `ns` variables, respectively

#### 5. The `zmin` and `zmax` variable corresopnd to the minimum and maximum Z-stack number to be included in analysis. For example, if you have a Z-stack with 10 focal planes and you only want to analyze the first 5, `zmin` would be 1 and `zmax` would be 5. If you want to analyze all 10 focal planes, assign `None` to both variables

#### 6. Understanding `z_multiplier`
proDiVis will generate orthogonal projections of your Z-stack, meaning you will view your Z-stack in x, y, and z planes, simultaneously. Increasing `z_multiplier` will multiply the pixels of your Z-stack in the z direction, meaning that the z plane will be stretched. This is useful if you want to visualize your SOI in the z direction.

#### 7. If you want to save your figures, assign `True` to the `save_figs` variable. If not, assign `False`. We reccomend running through the `Data_Pipeline.ipynb` notebook once without saving figures to ensure that the analysis is correct.

#### 8. Adjusting lower and upper thresholds
Normalization by proDiVis begins with histogram thresholding, a technique that segments an image by setting a range of pixel intensity values to be considered for analysis. The user is required to select a lower and upper boundary which correspond to the minimum and maximum pixel intensity values. ProDiVis excludes any pixel value outside of the user-defined boundaries. We reccomend setting starting with `lower_thresh = 0` and `upper_thresh = 254`. This will include all pixel values that are not saturated in the analysis/normalization

#### 9. Scaling images
You will reach a section of `Data_Pipeline.ipynb` that says "Image Rescaling". This section allows you to look at the normalized images and apply a scaling factor to brighten them, this step is optional. A graphical user interface will display in the file if you run `tools.stack_gui`.

If you choose to rescale images, they will be saved in a new directory adjacent to your directory that contains the input images.

In the next cell, the path to the scaled images will need to be typed and assigned to the `scaled_norm_stack_tiffs` variable (similar to step 4)


#### 10. The rest of the `Data_Pipeline.ipynb` file is designed to work seamlessly without the need for any more user input

