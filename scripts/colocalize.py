import os
import cv2
import matplotlib.pyplot as plt
import tools
import normalize
import numpy as np
import argparse
import sys
import norm_heatmap as nhmp
import warnings

def colocalizer(tiffsA, tiffsB, threshold):
    if len(tiffsA) != len(tiffsB):
        raise ValueError(f"First directory has {len(tiffsA)} files while the second has {len(tiffsB)}. They must have the same number of files.")

    tiffA_root = '_'.join(os.path.dirname(tiffsA[0]).split(os.path.sep)[-1].split('_')[:3])
    tiffB_root = '_'.join(os.path.dirname(tiffsB[0]).split(os.path.sep)[-1].split('_')[:3])
    new_dir = os.path.sep.join(tiffsA[0].split(os.path.sep)[:-2] + [f"{tiffA_root}_COL_{tiffB_root}_t{threshold}"])
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    else:
        use_temp = tools.get_int_input(f"Folder already found at {new_dir}. Should colocalization tiffs be regenerated (1-y; 0-n)? ", 0, 1)
        if not use_temp:
            return tools.get_files(new_dir)

    colocalized_tiffs = []
    i = 1
    for tA, tB in zip(tiffsA, tiffsB):
        print(f'Colocalizing {i}/{len(tiffsA)}', end = '\r')
        colocalized_tiffs.append(colocalize(tA, tB, new_dir, threshold, i-1, 5))
        i += 1
    print('')

    return colocalized_tiffs


def colocalize(tA, tB, out_dir, threshold, z, blur_size = 3):
    a_name = '_'.join('.'.join(tA.split(os.path.sep)[-1].split('.')[:-1]).split('_')[:-1])
    b_name = '_'.join('.'.join(tB.split(os.path.sep)[-1].split('.')[:-1]).split('_')[:-1])
    output = os.path.join(out_dir, f"col_{a_name}_{b_name}_z{z}.tiff")

    kernel = np.ones((blur_size, blur_size), np.float32) / (blur_size ** 2)
    tA_img = cv2.imread(tA)
    tA_bw = cv2.cvtColor(tA_img, cv2.COLOR_BGR2GRAY)
    tA_lp = cv2.filter2D(tA_bw, -1, kernel)
    tB_img = cv2.imread(tB)
    tB_bw = cv2.cvtColor(tB_img, cv2.COLOR_BGR2GRAY)
    tB_lp = cv2.filter2D(tB_bw, -1, kernel)

    if tA_bw.shape != tB_bw.shape:
        raise ValueError(f"Images {tA} and {tB} are not the same shape")

    min_matrix = np.zeros(tA_bw.shape)
    max_matrix = np.ones(tA_bw.shape) * 255

    threshold = threshold / (blur_size ** 2)
    colocalized = np.where((tA_lp >= threshold) & (tB_lp >= threshold), max_matrix, min_matrix)
    colocalized = colocalized.astype(np.uint8)
    cv2.imwrite(output, colocalized)
    return output


def main():
    # Creating command-line parser
    parser = argparse.ArgumentParser(description = 'Create a colocalization heatmap from two Z-stacks of the same tumoroid')
    parser.add_argument('signalA', help = 'Path to directory with stack of images of the signal from the first channel')
    parser.add_argument('signalB', help = 'Path to directory with stack of images of the signal from the second channel')
    parser.add_argument('-o', '--out', help = 'The directory where the heatmaps should be outputted', default = os.path.join(os.path.dirname(__file__), 'heatmaps'))
    parser.add_argument('-M', '--multiplier', help = 'The y-multiplier to thicken each slice in front and side views', default = 1)
    parser.add_argument('-Zs', '--zStart', help = 'The smallest z value to be used (counts up from 0)', default = 0) # TODO: Change Back (0)
    parser.add_argument('-Ze', '--zEnd', help = 'The largest z value to be used (cannot be higher than stack size)', default = -1) # TODO: Change Back (-1)
    parser.add_argument('-v', '--view', nargs = '*', help = 'Which heatmaps to generate; choose from x, y, and z', default = ['z'])
    parser.add_argument('-a', '--algorithm', help = 'How heatmaps will be generated - 0 indicates stacking (for normal computers), 1 indicates matrices (10gb+ RAM may be needed)', default = 0)
    parser.add_argument('-n', '--norm', help = 'The directory where the normalization stack can be found. If provided, -t must be provided along with -m OR -s', default = 0)
    parser.add_argument('-t', '--threshold', help = 'The pixel intensity threshold at which pixels should be considered when calculating intensity averages for normalization; REQUIRED when normalizing', default = '')
    parser.add_argument('-V', '--visualization', nargs = '*', help = 'Indicates what type of visualization should be generated. "m" for mean heatmap. "c" for cell boundary heatmap. "v" for 3D visualization', default = [])
    parser.add_argument('-b', '--cellBoundary', help = "Directory with cell boundary stack, typically a Phalloidin stain. REQUIRED with '-s'", default = 0)
    parser.add_argument('-p', '--prototxt', help = "Path to '.prototxt' file for use with edge detection CNN. REQUIRED with '-s'", default = '')
    parser.add_argument('-d', '--model', help = "Path to CNN model file for use with edge detection. REQUIRED with '-s'", default = '')
    args = parser.parse_args()

    # Ensuring provided directories are valid
    proteinA = tools.norm_dirname(args.signalA, 'tiff stack for protein signal A', False)
    proteinB = tools.norm_dirname(args.signalB, 'tiff stack for protein signal B', False)
    out_dir = tools.norm_dirname(args.out, 'output', True)
    norm_dir = tools.norm_dirname(args.norm, 'tiff stack for normalization', False)
    bound_dir = tools.norm_dirname(args.cellBoundary, 'tiff stack with cell boundary stain', False)

    # Extracting other variables
    viewpoints = args.view
    threshold = args.threshold
    z_multiplier = tools.smart_check_int_param(args.multiplier, 'multiplier', 1, 100)
    threshold = tools.smart_check_int_param(args.threshold, 'threshold', 0, 50)
    algorithm = tools.smart_check_int_param(args.algorithm, 'algorithm', 0, 1)

    # Getting all tiffs from the stack directory
    tiffsA = tools.get_files(proteinA)
    tiffsB = tools.get_files(proteinB)
    z_min = tools.smart_check_int_param(args.zStart, 'start of z stack bounds', 0, len(tiffsA) - 3)
    z_max = tools.smart_check_int_param(args.zEnd if args.zEnd != -1 else str(len(tiffsA)), 'end of the z stack bounds', z_min + 1, len(tiffsA))

    # Processing normalization information
    if norm_dir == 0:
        raise ValueError('Program call must include "-n" with directory that contains the normalization stack')

    for vis in args.visualization:
        norms = tools.get_files(norm_dir)
        prefix_tiffs = {}

        if vis == 'm':
            # Normalizing stack tiffs by normalization tiffs (if norm tiffs provided)
            tiffsAM = normalize.mean_normalizer(tiffsA, norms, threshold)
            tiffsBM = normalize.mean_normalizer(tiffsB, norms, threshold)
            # Prefix represents file prefix for generated heatmaps
            prefix = f'{proteinA.split(os.path.sep)[-2]}_col_{proteinB.split(os.path.sep)[-1]}_n_{norm_dir.split(os.path.sep)[-1]}_mean_t{threshold}_z{z_min}-{z_max}'
            prefix_tiffs[prefix] = [tiffsAM, tiffsBM]

        elif vis == 'c':
            phalloidins = tools.get_files(bound_dir)
            # TODO: Error trap prototxt and model file inputs
            tiffsAC = normalize.cell_normalizer(tiffsA, norms, phalloidins, args.prototxt, args.model, 10, (50, 50))
            tiffsBC = normalize.cell_normalizer(tiffsB, norms, phalloidins, args.prototxt, args.model, 10, (50, 50))
            prefix = f'{stack_dir.split(os.path.sep)[-2]}_{stack_dir.split(os.path.sep)[-1]}_n_{norm_dir.split(os.path.sep)[-1]}_cellSpecific_t{threshold}_z{z_min}-{z_max}'
            warnings.warn("Cell Specific normalization is not currently supported")
            # prefix_tiffs[prefix] = [tiffsAC, tiffsBC]

        elif vis == 'v':
            warnings.warn("Generation of 3D Visualization is not currently supported")

        else:
            raise ValueError(f"Could not understand visualization identifier {vis}. Use '--help' for more information")

    # Creating new directory within the output directory for the heatmaps with
    # the specific parameters provided in this run
    for prefix, tiffs in prefix_tiffs.items():
        out_dir = os.path.join(out_dir, prefix)
        tools.smart_make_dir(out_dir)

        tiffsA, tiffsB = tiffs
        tiffs = colocalizer(tiffsA, tiffsB, threshold)
        # Getting list of output heatmap image objects (pixel arrays)
        if algorithm == 1:
            out = nhmp.matrix_stack(tiffs[z_min:z_max], viewpoints, z_multiplier)
        else:
            out = nhmp.stack(tiffs[z_min:z_max], viewpoints, z_multiplier)

        nhmp.save_heatmaps(out_dir, prefix, viewpoints, out)


if __name__ == "__main__":
    main()
