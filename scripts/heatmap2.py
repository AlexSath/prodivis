import tools
import normalize
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import cv2

def create_matrix(tiff_list, thresh):
    l = []
    for tiff in tiff_list:
        l.append(cv2.cvtColor(cv2.imread(tiff), cv2.COLOR_BGR2GRAY))
    arr = np.asarray(l, float)
    arr[arr < thresh] = np.nan
    return arr


def main():
    # Creating command-line parser
    parser = argparse.ArgumentParser(description = 'Create a Basic Heatmap from an Image Stack')
    parser.add_argument('stack_dir', help = 'Path to directory with image stacks')
    parser.add_argument('-o', '--out', help = 'The directory where the heatmaps should be outputted', default = os.path.join(os.path.dirname(__file__), 'heatmaps'))
    parser.add_argument('-t', '--threshold', help = 'The pixel intensity threshold at which pixels should be considered when calculating intensity averages for normalization', default = 10)
    parser.add_argument('-M', '--multiplier', help = 'The y-multiplier to thicken each slice in front and side views', default = 20)
    parser.add_argument('-v', '--view', nargs = '*', help = 'Which heatmaps to generate; choose from x, y, and z', default = ['z'])
    parser.add_argument('-n', '--norm', help = 'The directory where the normalization stack can be found. If provided, -t must be provided along with -m OR -s', default = 0)
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
    z_multiplier = tools.smart_check_int_param(args.multiplier, 'multiplier', 1, 100)
    threshold = tools.smart_check_int_param(args.threshold, 'threshold', 0, 50)

    # Getting all tiffs from the stack directory
    tiffs = tools.get_files(stack_dir)
    objective = create_matrix(tiffs, threshold)

    # Processing normalization information
    if norm_dir:

        # Handling threshold command-line argument

        # Getting all tiffs from the normalization directory if it exists
        norms = tools.get_files(norm_dir)
        normalization = create_matrix(norms, threshold)

        # Handling thresholding types:
        if not args.mean and not args.cellSpecific:
            raise ValueError(f"When thresholding, '-m' or '-s' required. See '--help' for more information")
        elif args.mean and args.cellSpecific:
            raise ValueError(f"When thresholding, select either '-m' OR '-s', not both. See '--help' for more information")

        if args.mean:
            # Normalizing stack tiffs by normalization tiffs (if norm tiffs provided)
            tiffs = normalize.matrix_mean_normalizer(objective, normalization)
            # Prefix represents file prefix for generated heatmaps
            prefix = f'{stack_dir.split(os.path.sep)[-2]}_{stack_dir.split(os.path.sep)[-1]}_norm_{norm_dir.split(os.path.sep)[-1]}_mean_t{threshold}'
        elif args.cellSpecific:
            phalloidins = tools.get_files(bound_dir)
            # TODO: Error trap prototxt and model file inputs
            tiffs = normalize.cell_normalizer(tiffs, norms, phalloidins, args.prototxt, args.model, 10, (50, 50))
            prefix = f'{stack_dir.split(os.path.sep)[-2]}_{stack_dir.split(os.path.sep)[-1]}_norm_{norm_dir.split(os.path.sep)[-1]}_cellSpecific_t{threshold}'
            sys.exit("Further processing of cell Specific boundaries is not supported")
        else:
            raise ValueError("Could not determine whether normalization was 'mean' or 'cellSpecific'")

    else:
        prefix = f'{stack_dir.split(os.path.sep)[-1]}'

    # Creating new directory within the output directory for the heatmaps with
    # the specific parameters provided in this run
    out_dir = os.path.join(out_dir, prefix)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # Getting list of output heatmap image objects (pixel arrays)
    out = stack(tiffs, viewpoints, z_multiplier)
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
