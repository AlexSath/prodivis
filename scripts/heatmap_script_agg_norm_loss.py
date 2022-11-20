import norm_heatmap
import normalize
import experiment_normalized_depth_loss
import tools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import heatmap_workflow_tools as hwt
import os

# In quotes, put paths to folders containing tiffs for SOI and Normalization signals, respectively
stack_dir = os.path.abspath("/Users/Kyle/Desktop/Heat Map Code/Heatmaps/Image_7_COXI_COXII/COXI")
norm_dir = os.path.abspath("/Users/Kyle/Desktop/Heat Map Code/Heatmaps/Image_7_COXI_COXII/DAPI/")

# Replace with slice integers or decimal percentages. 'None' for no boundary.
zmin, zmax = None, None

# how many microns is a single image/slice?
slice_depth = 0.79

# how many slices does a monolayer of your cell type represent?
monolayer_num_slices = 15

# set the lower threshold of your pixel intensity that you want to collect

# do you want to report depth loss as average pixel intensity per slice or median pixel intensity per slice?
plot_median = True 
plot_mean = True   
show_stds = True

# do you want to normalize by the mean pixel intensity or median pixel intensity?


# do you want to save your figures?
save_figs = False 

stack_tiffs = tools.get_files(stack_dir)
norm_tiffs = tools.get_files(norm_dir)
zmin, zmax = hwt.process_zmin_zmax(zmin, zmax, stack_tiffs)
#print(f"Processed ZMin and ZMax: {zmin}, {zmax}")

stack_name = os.path.basename(stack_dir)
norm_name = os.path.basename(norm_dir)
base_dir = os.path.basename(os.path.dirname(norm_dir))
out_dir = os.path.abspath(os.path.join(os.path.abspath(''), '..', 'hmp_paper_data'))
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
out_dir = os.path.join(out_dir, f"{base_dir}_{stack_name}n{norm_name}_out")
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

    stack_means = np.array([])
stack_stds = np.array([])
stack_medians = np.array([])
for img in stack_tiffs[zmin:zmax]:
    mean, std, median = normalize.tiff_mean_std(img, 0, -1, True, 0)
    stack_means = np.append(stack_means, mean)
    stack_stds = np.append(stack_stds, std)
    stack_medians = np.append(stack_medians, median)
hwt.plot_MSI(stack_means, stack_stds, stack_medians, slice_depth,
             f"{base_dir} {stack_name} z{zmin}-{zmax}",
             os.path.join(out_dir, f"{base_dir}_{stack_name}_z{zmin}-{zmax}_"), plot_median, plot_mean, show_stds, save_figs)

norm_means = np.array([])
norm_stds = np.array([])
norm_medians = np.array([])
for norm_tiff in norm_tiffs[zmin:zmax]:
    mean, std, medians = normalize.tiff_mean_std(norm_tiff, 0, -1, True, 0)
    norm_means = np.append(norm_means, mean)
    norm_stds = np.append(norm_stds, std)
    norm_medians = np.append(norm_medians, median)
hwt.plot_MSI(norm_means, norm_stds, norm_medians, slice_depth,
             f"{base_dir} {norm_name} z{zmin}-{zmax}",
             os.path.join(out_dir, f"{base_dir}_{stack_name}_z{zmin}-{zmax}_"), plot_median, plot_mean, show_stds, save_figs)
