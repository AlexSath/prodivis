# ProDiVis
A Z-stack validation suite written in the python programming language, designed to be user-friendly for those with little knowledge of image analysis or programming. ProDiVis takes Z-stack outputs acquired from bioformat files (for example: CARL ZEISS .czi or Leica .lif) and generates a heatmap of differentially localized protein(s) normalized to a user-selected fluorescent housekeeping signal. Heatmaps can be generated from any fluorescent Z-tack and serve as an unbiased visualization tool. In addition to producing heatmaps, proDiVis outputs a normalized Z-stack that can be readily used with concurrent microscopy images. Built for data acquired in 'z-stack' form, where different optical sections are imaged individually.

## Inherent Problems with Confocal Deep Imaging
A common problem with laser scanning confocal microscopy is the intensity of excitation light and emission signal from fluorophores decrease as they travel through 3D specimens, resulting in weaker signal reaching the detector. To facilitate image interpretation, we developed proDiVis: a visualization algorithm involving focal-plane-specific signal normalization. However, there are limitations to optical sectioning while imaging thick specimens, namely fluorescence intensity loss as a function of imaging depth. The most widely used fluorophores have emission light in the visible spectrum that can only penetrate a limited distance through biological material. This is a known contributor to the fundamental depth limit, caused by several physical properties such as light scattering and absorption. To address the above issues, we developed a computational method to proportionally compare pixel values across the depth of 3D specimens, accounting for the decrease in fluorescence intensity we and others have previously observed. While there are many existing programs or software available for image analysis, to the best of our knowledge, none of them account for loss of signal by normalizing against a housekeeping signal, and many of them are expensive and/or require a high level of technical expertise to use.

# Quickstart Guide (see below for more detailed instructions for novice users)
## Installation Instructions
Using pip

'''
pip install -r requirements.txt
'''

Using conda
'conda install --file requirements.txt'

"Data_Pipeline.ipynb" contains all the necessary code to run proDiVis

# Detailed Installation and Operation Instructions for Novice Users
We reccomend using python 3.11.3 (latest release as of April 5, 2013) which can be found
[here](https://www.python.org/downloads/)

We run proDiVis and subsequently python through [Anaconda](https://www.anaconda.com/), which is a software management tool designed to simplify the process of working with python and many python packages required for scientific computing. Using either method of installation works.

navigate to the directory (folder) where you wish to install proDiVis
using the command line
'git clone  https://github.com/AlexSath/prodivis.git'
you should have downloaded all the files needed to run proDiVis inside this folder

## using pip to create en environment
create a virtual environment inside the directory that proDiVis is located in, replace 'env_name' with the desired name of your virtual environment.
'python -m venv env_name'
activate the environment 
windows: '.\env_name/Scripts\activate
macOS: source env_name/bin/activate
install required packages
'pip install -r requirements.txt'

## using conda to create an environment


