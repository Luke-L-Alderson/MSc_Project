from helpers import umap_plt, dtype_transform
import seaborn as sns
import pandas as pd
import matplotlib as mpl
from matplotlib import cm, colors
from matplotlib import pyplot as plt
import tonic
import tonic.transforms as transforms
import torch
import snntorch.spikeplot as splt
from IPython.display import HTML
import numpy as np
# # recurrency
# tsne_plt("75 Hz_1 Hz_0")
# tsne_plt("75 Hz_1 Hz_1")

# Rate Changes (Heat Map)
# umap_plt("datafiles/75 Hz_1 Hz_0.csv", 15, 10)
# umap_plt("datafiles/75 Hz_0.75 Hz_0.csv", 15, 10)

# Heatmap (Seaborn)
# labels = ["1", "50", "100", "150", "200"]
# labels = [1, 50, 100, 150, 200]
# hmap = pd.read_csv("rates heatmap data.csv", index_col=0)#, header=labels, index_col=labels)

# plt.figure(figsize = (10,7))
# plt.title('Rates Heat Map (Errors normalised by frequency)')
# ax = sns.heatmap(hmap*100, vmin=0.0, vmax=40, cmap = 'cividis', annot = True, fmt = '.2f')
# plt.xlabel('Rate On, Hz')
# plt.ylabel('Rate Off, Hz')
# ax.invert_yaxis()
# ax.set_facecolor('y')
# plt.savefig('heatmap.png')
# plt.savefig('heatmap.pdf')

# NMNIST Input Data




dataset = tonic.datasets.NMNIST(save_to='./dataset',
                                      train=True,
                                      first_saccade_only=False
                                      )
targets = set()
for events, target in dataset:
    if target not in targets:
        targets.add(target)
        tonic.utils.plot_event_grid(events)
        if len(targets) == 3:
            break

sensor_size = tonic.datasets.NMNIST.sensor_size
print(f"Sensor Size: {sensor_size}")
raw_transform = tonic.transforms.Compose([
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
            torch.from_numpy,
            dtype_transform()
            ])

dataset = tonic.datasets.NMNIST(save_to='./dataset',
                                      train=True,
                                      first_saccade_only=False,
                                      transform = raw_transform
                                      )

print(f"Dataset has {len(dataset)} elements.")
targets = set()
im_list = []
for events, target in dataset:
    if target not in targets:
        print(target)
        print(events.shape)
        targets.add(target)
        im_list.append(events)
        
        ax1 = plt.subplot(1, 3, 1)
        plt.imshow(im_list[target][1].squeeze())
        plt.xlabel("t = 1")

        ax2 = plt.subplot(1, 3, 2)
        plt.imshow(im_list[target][50].squeeze())
        plt.xlabel("t = 100")
        ax3 = plt.subplot(1, 3, 3)
        plt.imshow(im_list[target][-1].squeeze())
        plt.xlabel("t = 300")
        if len(targets) == 3:
            break



times = []
for i, (events, target) in enumerate(dataset):
    print(f"{i}/{len(dataset)}")    
    targets.add(target)
    times.append(events.shape[0])

print(np.array(times).mean())
print(np.array(times).std())