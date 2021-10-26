import os
import cv2
import matplotlib.pyplot as plt
import tools
import normalize
import numpy as np
import argparse
import sys
# import cython

""" DEPRECATED:
# def get_gscale_images(tiff_list):
#     img_list = []
#     for tiff in tiff_list:
#         img = cv2.imread(tiff)
#         bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img_list.append(bw)
#     return img_list

# def top_stack(tiff_list):
#     shape = cv2.cvtColor(cv2.imread(tiff_list[0]), cv2.COLOR_BGR2GRAY).shape
#     stack_size = len(tiff_list)
#     agg_img = np.zeros(shape)
#     for tiff in tiff_list:
#         img = cv2.imread(tiff)
#         bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         if bw.shape != agg_img.shape:
#             raise ValueError("Tiffs in the provided folder not all the same size. Cannot compile composite heatmap.")
#         agg_img += bw
#     return agg_img / stack_size
"""

def create_matrix(tiff_list):
    l = []
    for tiff in tiff_list:
        l.append(cv2.cvtColor(cv2.imread(tiff), cv2.COLOR_BGR2GRAY))
    arr = np.asarray(l, float)
    return arr

def mult_view(img, z_mult):
    out = []
    for row in img:
        out.extend(np.tile(row, (z_mult,1)))
    out = np.asarray(out)
    return out

def matrix_stack(tiff_list, viewpoints, z_mult):
    print(f"Loading data into matrix...")
    A = create_matrix(tiff_list)
    img_list = []
    for viewpoint in viewpoints:
        print(f"Generating heatmap for viewpoint {viewpoint}")
        if viewpoint == 'z':
            img_list.append(tools.min_max_scale(np.nanmean(A, axis = 0)))
        elif viewpoint == 'y':
            img_list.append(tools.min_max_scale(mult_view(np.nanmean(A, axis = 1), z_mult)))
        elif viewpoint == 'x':
            img_list.append(tools.min_max_scale(mult_view(np.nanmean(A, axis = 2), z_mult)))
    return img_list



# Function: stack()
# Description: Stacks provided tiffs into composite image from each of the views
#              given. 'x', 'y', and 'z' are valid views, representing the 3D axes
#              of the image composite.
# Pre-Conditions: Provided list of path-like strings to tiff files, list of desired
#                 viewpoints, integer multiplier for widening of z-axis on x and y
#                 views, and boolean 'norm' (currently non-functional)
# Post-Conditions: List of composite image objects generated from each view.
def stack(tiff_list, viewpoints, z_mult, norm = False):
    shape = cv2.cvtColor(cv2.imread(tiff_list[0]), cv2.COLOR_BGR2GRAY).shape
    stack_size = len(tiff_list)
    img_list = []
    for viewpoint in viewpoints:
        if viewpoint == 'z':
            img_list.append(np.zeros(shape))
        elif viewpoint == 'y':
            img_list.append(np.zeros((stack_size, shape[0])))
        elif viewpoint == 'x':
            img_list.append(np.zeros((stack_size, shape[1])))
        else:
            raise ValueError(f"Viewpoint {viewpoint} not understood. Please choose from x, y, or z")

    print(f"Generating composites for {', '.join(viewpoints[:-1])} and {viewpoints[-1] if len(viewpoints) > 1 else viewpoints[0]} view(s) from {os.path.dirname(tiff_list[0])}...")
    for idx, tiff in enumerate(tiff_list):
        img = cv2.imread(tiff)
        bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if bw.shape != shape:
            raise ValueError("Tiffs in the provided folder not all the same size. Cannot compile composite heatmap.")

        if 'x' in viewpoints or 'y' in viewpoints:
            transposed = cv2.transpose(bw)

        for i in range(len(viewpoints)):
            if viewpoints[i] == 'z':
                img_list[i] += bw
            elif viewpoints[i] == 'y':
                for jdx, row in enumerate(bw):
                    img_list[i][idx][jdx] = np.mean(row)
            elif viewpoints[i] == 'x':
                for jdx, row in enumerate(transposed):
                    img_list[i][idx][jdx] = np.mean(row)


    print(f"Completing post-processing for {', '.join(viewpoints[:-1])} and {viewpoints[-1] if len(viewpoints) > 1 else viewpoints[0]} composite(s) from {os.path.dirname(tiff_list[0])}...")
    for idx, viewpoint in enumerate(viewpoints):
        if viewpoint == 'z':
            img_list[idx] /= stack_size
        else:
            out = np.zeros((img_list[idx].shape[0] * z_mult, img_list[idx].shape[1]))
            for rdx, rows in enumerate(out):
                for cdx, cols in enumerate(rows):
                    og_row = int(np.floor(rdx / z_mult))
                    out[rdx][cdx] = img_list[idx][og_row][cdx]
            img_list[idx] = out
        img_list[idx] = tools.min_max_scale(img_list[idx])

    return img_list


def main():
    # Creating command-line parser
    parser = argparse.ArgumentParser(description = 'Create a Basic Heatmap from an Image Stack')
    parser.add_argument('stack_dir', help = 'Path to directory with image stacks')
    parser.add_argument('-o', '--out', help = 'The directory where the heatmaps should be outputted', default = os.path.join(os.path.dirname(__file__), 'heatmaps'))
    parser.add_argument('-M', '--multiplier', help = 'The y-multiplier to thicken each slice in front and side views', default = 1)
    parser.add_argument('-Zs', '--zStart', help = 'The smallest z value to be used (counts up from 0)', default = 0)
    parser.add_argument('-Ze', '--zEnd', help = 'The largest z value to be used (cannot be higher than stack size)', default = -1)
    parser.add_argument('-v', '--view', nargs = '*', help = 'Which heatmaps to generate; choose from x, y, and z', default = ['z'])
    parser.add_argument('-n', '--norm', help = 'The directory where the normalization stack can be found. If provided, -t must be provided along with -m OR -s', default = 0)
    parser.add_argument('-t', '--threshold', help = 'The pixel intensity threshold at which pixels should be considered when calculating intensity averages for normalization; REQUIRED when normalizing', default = '')
    parser.add_argument('-m', '--mean', help = "Indicator that the 'mean' method for normalization should be used. MUTUALLY EXCLUSIVE with '-s'", action = 'store_true')
    parser.add_argument('-s', '--cellSpecific', help = "Indicator that the 'cell-specific' method for normalization should be used. MUTUALLY EXCLUSIVE with '-m'. REQUIRES '-b'.", action = 'store_true')
    parser.add_argument('-b', '--cellBoundary', help = "Directory with cell boundary stack, typically a Phalloidin stain. REQUIRED with '-s'", default = 0)
    parser.add_argument('-p', '--prototxt', help = "Path to '.prototxt' file for use with edge detection CNN. REQUIRED with '-s'", default = '')
    parser.add_argument('-d', '--model', help = "Path to CNN model file for use with edge detection. REQUIRED with '-s'", default = '')
    args = parser.parse_args()

    # Ensuring provided directories are valid
    stack_dir = tools.norm_dirname(args.stack_dir, 'tiff stack for heatmap', False)
    out_dir = tools.norm_dirname(args.out, 'output', True)
    norm_dir = tools.norm_dirname(args.norm, 'tiff stack for normalization', False)
    bound_dir = tools.norm_dirname(args.cellBoundary, 'tiff stack with cell boundary stain', False)

    # Extracting other variables
    viewpoints = args.view
    threshold = args.threshold
    z_multiplier = tools.smart_check_int_param(args.multiplier, 'multiplier', 1, 100)
    threshold = tools.smart_check_int_param(args.threshold, 'threshold', 0, 50)

    # Getting all tiffs from the stack directory
    tiffs = tools.get_files(stack_dir)
    z_min = tools.smart_check_int_param(args.zStart, 'start of z stack bounds', 0, len(tiffs) - 3)
    z_max = tools.smart_check_int_param(args.zEnd if args.zEnd != -1 else str(len(tiffs)), 'end of the z stack bounds', z_min + 1, len(tiffs))

    # Processing normalization information
    if norm_dir:
        # Getting all tiffs from the normalization directory if it exists
        norms = tools.get_files(norm_dir)

        # Handling thresholding types:
        if not args.mean and not args.cellSpecific:
            raise ValueError(f"When thresholding, '-m' or '-s' required. See '--help' for more information")
        elif args.mean and args.cellSpecific:
            raise ValueError(f"When thresholding, select either '-m' OR '-s', not both. See '--help' for more information")

        if args.mean:
            # Normalizing stack tiffs by normalization tiffs (if norm tiffs provided)
            tiffs = normalize.mean_normalizer(tiffs, norms, threshold)
            # Prefix represents file prefix for generated heatmaps
            prefix = f'{stack_dir.split(os.path.sep)[-2]}_{stack_dir.split(os.path.sep)[-1]}_n_{norm_dir.split(os.path.sep)[-1]}_mean_t{threshold}_z{z_min}-{z_max}'
        elif args.cellSpecific:
            phalloidins = tools.get_files(bound_dir)
            # TODO: Error trap prototxt and model file inputs
            tiffs = normalize.cell_normalizer(tiffs, norms, phalloidins, args.prototxt, args.model, 10, (50, 50))
            prefix = f'{stack_dir.split(os.path.sep)[-2]}_{stack_dir.split(os.path.sep)[-1]}_n_{norm_dir.split(os.path.sep)[-1]}_cellSpecific_t{threshold}_z{z_min}-{z_max}'
            sys.exit("Further processing of cell Specific boundaries is not supported")
        else:
            raise ValueError("Could not determine whether normalization was 'mean' or 'cellSpecific'")

    else:
        prefix = f'{stack_dir.split(os.path.sep)[-1]}_z{z_min}-{z_max}'

    # Creating new directory within the output directory for the heatmaps with
    # the specific parameters provided in this run
    out_dir = os.path.join(out_dir, prefix)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # Getting list of output heatmap image objects (pixel arrays)
    # out = matrix_stack(tiffs[z_min:z_max], viewpoints, z_multiplier)
    out = stack(tiffs[z_min:z_max], viewpoints, z_multiplier)

    # Using matplotlib to generate a heatmap for every image object generated.
    for v, o in zip(viewpoints, out):
        output_file = os.path.join(out_dir, f'{prefix}_{v}.tif')
        print(f"Saving heatmap '{output_file}'...")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(o, cmap = 'magma', interpolation = 'nearest')
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(output_file, format='tif', dpi = 1200, bbox_inches = 'tight')


    # If normalization was performed, give the user the option to delete the generated normalized tiffs (to save space)
    if norm_dir != '':
        delete = tools.get_int_input(f"Would you like to delete the Normalized tiff folder {os.path.dirname(tiffs[0])} (1-y; 0-n)? ", 0, 1)
        if delete:
            print(f"Removing {os.path.dirname(tiffs[0])}...")
            os.rmdir(os.path.dirname(tiffs[0]))


if __name__ == "__main__":
    main()
