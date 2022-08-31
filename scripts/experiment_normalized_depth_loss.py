import tools as t
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import sys
import os

def graph_normalized_depths(csv_in, outpath):
    csv_name = csv_in.split(os.path.sep)[-1].split('.')[0]
    t.smart_make_dir(os.path.join(outpath, csv_name))
    df = pd.read_csv(csv_in)
    for col in df.columns[1:]:
        col_data = np.array(df[col])
        nanidxs = np.where(np.isnan(col_data))
        col_data = col_data[:nanidxs[0][0]] if len(nanidxs[0]) != 0 else col_data
        col_data /= np.max(col_data)
        xs = np.arange(1, len(col_data) + 1) * 0.79
        ys = col_data * 100
        plt.plot(xs, ys)
        plt.xlabel("Tumor Depth (\u03BCm)")
        plt.ylabel("% Maximum Signal")
        plt.ylim([0, np.max(ys)])
        plt.title(f"{csv_name.replace('_', ' ')}: {col.replace('_', ' ')}")
        plt.savefig(os.path.join(outpath, csv_name, f"{csv_name}_{col}.png"))
        plt.close()


def main():
    filedir = sys.argv[1]
    if not os.path.isdir(filedir):
        raise ValueError("Provided path is not a directory!")
    keyword = sys.argv[2]
    filepaths = t.get_keyword_files(filedir, keyword)
    outdir = sys.argv[3]
    t.smart_make_dir(outdir)
    outdir = os.path.join(outdir, "exp_specific")
    t.smart_make_dir(outdir)
    for file in filepaths:
        graph_normalized_depths(file, outdir)


if __name__ == '__main__':
    main()
