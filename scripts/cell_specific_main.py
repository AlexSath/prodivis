"""
Filename: cell_specific_main.py
Author: Alexandre R. Sathler
Date: 05/04/2022
Description: Storage file for old code containing conditionals to process whether
mean or cell-specific normalization should be performed. If cell-specific normalization
and / or 3D visualization is developed in the future, this may be needed.
"""

def main():

    for vis in args.visualization:
        norms = tools.get_files(norm_dir)
        prefix_tiffs = {}

        if vis == 'm':
            # Normalizing stack tiffs by normalization tiffs (if norm tiffs provided)
            tiffsM = normalize.mean_normalizer(tiffs, norms, threshold)
            # Prefix represents file prefix for generated heatmaps
            prefix = f'{stack_dir.split(os.path.sep)[-2]}_{stack_dir.split(os.path.sep)[-1]}_n_{norm_dir.split(os.path.sep)[-1]}_mean_t{threshold}_z{z_min}-{z_max}'
            prefix_tiffs[prefix] = tiffsM

        elif vis == 'c':
            phalloidins = tools.get_files(bound_dir)
            # TODO: Error trap prototxt and model file inputs
            tiffsC = normalize.cell_normalizer(tiffs, norms, phalloidins, args.prototxt, args.model, 10, (50, 50))
            prefix = f'{stack_dir.split(os.path.sep)[-2]}_{stack_dir.split(os.path.sep)[-1]}_n_{norm_dir.split(os.path.sep)[-1]}_cellSpecific_t{threshold}_z{z_min}-{z_max}'
            warnings.warn("Cell Specific normalization is not currently supported")
            # prefix_tiffs[prefix] = tiffsC

        elif vis == 'v':
            warnings.warn("Generation of 3D Visualization is not currently supported")

        else:
            raise ValueError(f"Could not understand visualization identifier {vis}. Use '--help' for more information")

    
    # Creating new directory within the output directory for the heatmaps with
    # the specific parameters provided in this run
    for prefix, tiffs in prefix_tiffs.items():
        out_dir = os.path.join(out_dir, prefix)
        tools.smart_make_dir(out_dir)

        # Getting list of output heatmap image objects (pixel arrays)
        if algorithm == 1:
            out = matrix_stack(tiffs[z_min:z_max], viewpoints, z_multiplier)
        else:
            out = stack(tiffs[z_min:z_max], viewpoints, z_multiplier)

        save_heatmaps(out_dir, prefix, viewpoints, out)
