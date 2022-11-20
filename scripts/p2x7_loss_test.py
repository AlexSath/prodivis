import matplotlib as plt
import pandas as pd

DAPI = pd.read_csv('/Users/Kyle/Desktop/Heat Map Code/Heatmaps/Image_P2X7/ImageJ_DAPI_Measure.csv')
P2X7 = pd.read_csv('/Users/Kyle/Desktop/Heat Map Code/Heatmaps/Image_P2X7/ImageJ_P2x7_Measure.csv')

plt.plot(0, len(mean), mean)