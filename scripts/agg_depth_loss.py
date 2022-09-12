import os
import tools
import normalize
import argparse
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def get_folders(rootdir, names, inverse):
    dirs_found = []
    for root, dirs, files in os.walk(rootdir):
        for d in dirs:
            if not inverse and (d in names or d == names):
                dirs_found.append(os.path.join(root, d))
            elif inverse and d not in names and d != names:
                firstfile = os.listdir(os.path.join(root, d))[0]
                if '.tif' in firstfile and 'norm' not in firstfile and 'projection' not in firstfile:
                    dirs_found.append(os.path.join(root, d))
    return dirs_found

def agg_means(dirs, threshold, stddev, ignore_monolayer, cutoff = 0.75):
    meandict = {}
    for idx, d in enumerate(dirs):
        tiffs = tools.get_files(d)
        these_means = []
        norm_bool = 0 if not ignore_monolayer else normalize.get_norm_bool_idxs(tiffs, cutoff)
        for t in tiffs:
            these_means.append(normalize.tiff_mean(t, norm_bool, threshold, stddev, norm_bool))
        key = d.split(os.path.sep)[-2]
        meandict[key] = these_means
    maximum = np.max([len(x) for x in meandict.values()])
    for key, value in meandict.items():
        for i in np.arange(maximum - len(value)):
            meandict[key].append(np.nan)
    return pd.DataFrame(meandict)

def graph(df, outdir, names, inverse, fileout):
    xvals = df.index * 0.79
    plt.plot(xvals, df.minout, label = 'Med +/- 1.5IQR', color = 'red')
    plt.plot(xvals, df.maxout, color = 'red')
    plt.plot(xvals, df.means, label = 'Mean', color = 'forestgreen', linewidth = 2)
    plt.plot(xvals, df.med, label = 'Median', color = 'blue', linewidth = 2)
    ax = plt.gca()
    ax.fill_between(xvals, df.maxout, df.minout, color = 'red', alpha = 0.15)
    ax.set_title(f"{'Not ' if inverse else ''}{', '.join(names)} intensity loss")
    ax.set_xlabel('Tumor Depth (um)')
    ax.set_ylabel(' Slice Intensity')
    if ax.get_ylim()[0] < 0:
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], linestyle = '--', color = 'black')
    else:
        ax.set_ylim([0, ax.get_ylim()[1]])
    plt.legend()
    plt.savefig(fileout)


def main():
    parser = argparse.ArgumentParser(description = 'Get graph of intensity loss from identical stains.' + \
             '\nNOTE: Requires that identical stains are stored in directories with the same names.')
    parser.add_argument('rootdir', help = 'Directory where all identical stains are stored. May be nested in different folders.')
    parser.add_argument('-s', '--stacks', nargs = '*', help = 'Name of stack(s) to aggregate depth losses for.')
    parser.add_argument('-n', '--not-named', nargs = '?', help = 'When added, will aggregate depth losses for stacks not named', default = False, const = True)
    parser.add_argument('-o', '--output', help = 'Output directory for graphs')
    parser.add_argument('-t', '--threshold', help = 'Pixel threshold for normalized values', default = 0)
    parser.add_argument('-O', '--outlierHandling', help = "Pixels above '-O' standard deviations are not considered", default = -1)
    parser.add_argument('-im', '--ignore-monolayer', nargs = '?', help = 'Add this option when tumor boundaries should be calculated to exclude monolayer signal in normalization slice means', default = False, const = True)
    # parser.add_argument('-imc', '--ignore-monolayer', help =
    args = parser.parse_args()

    threshold = tools.smart_check_int_param(args.threshold, 'threshold', 0, 50)
    n_stddevs = -1 if args.outlierHandling == -1 else tools.smart_check_int_param(args.outlierHandling, 'number of standard deviations', 1, 7)
    outdir = tools.norm_dirname(args.output, 'output', True)
    datadir = tools.norm_dirname(os.path.join(outdir, 'data'), 'data', True)
    root = f"{'Not_' if args.not_named else ''}{'-'.join(args.stacks)}_intensity_loss{'' if threshold == 0 else f'_t{threshold}'}" \
           + f"{'' if n_stddevs == -1 else f'_{n_stddevs}stddev'}"
    imgout = os.path.join(outdir, root)
    dataout = os.path.join(datadir, root)

    rawdf_out = dataout + "_raw.csv"
    if os.path.isfile(rawdf_out):
        data_df = pd.read_csv(rawdf_out)
    else:
        dirs = get_folders(args.rootdir, args.stacks, args.not_named)
        data_df = agg_means(dirs, threshold, n_stddevs, args.ignore_monolayer, 0.60)
        data_df.to_csv(rawdf_out)

    analdf_out = dataout + "_anal.csv"
    if os.path.isfile(analdf_out):
        anal_df = pd.read_csv(analdf_out)
    else:
        anal_df = pd.DataFrame()
        anal_df['means'] = data_df.mean(axis = 1)
        anal_df['med'] = data_df.median(axis = 1)
        anal_df['iqr'] = data_df.quantile(0.75, axis = 1) - data_df.quantile(0.25, axis = 1)
        anal_df['minout'] = data_df.quantile(0.25, axis = 1) - 1.5 * anal_df.iqr
        anal_df['maxout'] = data_df.quantile(0.75, axis = 1) + 1.5 * anal_df.iqr
        anal_df.to_csv(analdf_out)

    graph(anal_df, args.output, args.stacks, args.not_named, imgout + ".png")

if __name__ == '__main__':
    main()
