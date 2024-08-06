from helpers import umap_plt, dtype_transform, build_network, PoissonTransform, set_seed, SignTransform
import seaborn as sns
import pandas as pd
import matplotlib as mpl
from matplotlib import cm, colors
from matplotlib import pyplot as plt
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import snntorch.spikeplot as splt
from IPython.display import HTML
import numpy as np
from torchvision.transforms import v2
from torchvision import datasets
from brian2 import *
import os
from umap import UMAP
import warnings
from glob import glob
import latex
from scipy import stats
# plt.rcParams.update({
#     "text.usetex": True
# })
# mpl.rcParams['text.usetex'] = True

def extract_numbers(file_name):
    base_name = file_name.split('.')[0]  # Remove the extension
    parts = base_name.split('_')  # Split by the underscore
    return int(parts[0]), int(parts[1])



# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# set_seed()
# cm = 1/2.54
# input_specs = {}
# input_specs["rate_on"] = 75*Hz
# input_specs["rate_off"] = 1*Hz    
# input_specs["total_time"] = 100*ms
# input_specs["bin_size"] = 1*ms
# total_time = input_specs["total_time"]
# bin_size = input_specs["bin_size"]
# rate_on = input_specs["rate_on"]
# rate_off = input_specs["rate_off"]

# transform_s = v2.Compose([
#             v2.Grayscale(),
#             v2.ToTensor(),
#             v2.Normalize((0,), (1,)),
#             PoissonTransform(total_time, bin_size, rate_on, rate_off)
#             ])

# transform_ns = v2.Compose([
#             v2.Grayscale(),
#             v2.ToTensor(),
#             v2.Normalize((0,), (1,)),
#             ])

# dataset_s = datasets.MNIST(root='dataset/', train=True, transform=transform_s, download=True)
# dataset_ns = datasets.MNIST(root='dataset/', train=True, transform=transform_ns, download=True)
# network, network_params = build_network(device, noise=0, recurrence=False, num_rec=100, size=28, kernel_size=7)

# ''' ENCODED MNIST DIGITS '''
# seen_labels = set()
# unique_ims = []
# orig_ims = []
# for i, (image, label) in enumerate(dataset_s):
#     print(f"{i+1}/{len(dataset_s)}")
#     if label not in seen_labels:
#         seen_labels.add(label)
#         orig_ims.append((image, label))
#         if len(seen_labels) == 10:
#             break

# orig_ims.sort(key=lambda x: x[1])

# #figsize=(12, 10)
# fig, axs = plt.subplots(1, 10, layout="constrained")

# # Flatten the axis array for easier indexing
# axs = axs.flatten()    
# for i in range(10):
#     axs[i].imshow(orig_ims[i][0].mean(0).squeeze(), cmap='grey')
#     axs[i].axis('off')      

# plt.tight_layout(pad=0, h_pad=0, w_pad=0)
   
# fig.savefig("figures/spiking_digits.png", bbox_inches="tight", pad_inches=0)

# ''' ORIGINAL MNIST DIGITS '''
# seen_labels = set()
# unique_ims = []
# orig_ims = []
# for i, (image, label) in enumerate(dataset_ns):
#     print(f"{i+1}/{len(dataset_ns)}")
#     if label not in seen_labels:
#         seen_labels.add(label)
#         orig_ims.append((image, label))
#         if len(seen_labels) == 10:
#             break

# orig_ims.sort(key=lambda x: x[1])

# #figsize=(12, 10)
# fig, axs = plt.subplots(1, 10, layout="constrained")

# # Flatten the axis array for easier indexing
# axs = axs.flatten()    
# for i in range(10):
#     axs[i].imshow(orig_ims[i][0].squeeze(), cmap='grey')
#     axs[i].axis('off')      


# plt.tight_layout(pad=0, h_pad=0, w_pad=0)
   
# fig.savefig("figures/nonspiking_digits.png", bbox_inches="tight", pad_inches=0)


''' ENCODING FIGURE '''
# t1, t2, t3 = 10, 50, 80
# index = 0
# inputs = dataset_s[index][0].squeeze()
# inputs_original = dataset_ns[index][0].squeeze()

# fig =  plt.figure(figsize=(10,2))

# ax1 = plt.subplot(1, 5, 1)
# ax1.imshow(inputs_original, cmap='grey')
# plt.xlabel("Original")
# plt.tick_params(
#     axis='both',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=True) # labels along the bottom edge are off
# ax1.set_xticks([])
# ax1.set_yticks([])

# ax2 = plt.subplot(1, 5, 2)
# ax2.imshow(inputs[t1], cmap='grey')
# plt.xlabel(f"t = {t1}")
# plt.tick_params(
#     axis='both',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=True) # labels along the bottom edge are off
# ax2.set_xticks([])
# ax2.set_yticks([])

# ax3 = plt.subplot(1, 5, 3)
# ax3.imshow(inputs[t2], cmap='grey')
# plt.xlabel(f"t = {t2}")
# plt.tick_params(
#     axis='both',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=True) # labels along the bottom edge are off
# ax3.set_xticks([])
# ax3.set_yticks([])

# ax4 = plt.subplot(1, 5, 4)
# ax4.imshow(inputs[t3], cmap='grey')
# plt.xlabel(f"t = {t3}")
# plt.tick_params(
#     axis='both',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=True) # labels along the bottom edge are off
# ax4.set_xticks([])
# ax4.set_yticks([])

# ax5 = plt.subplot(1, 5, 5)
# ax5.imshow(inputs.mean(axis=0), cmap='grey')

# plt.xlabel("Time Averaged")
# plt.tick_params(
#     axis='both',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=True) # labels along the bottom edge are off
# ax5.set_xticks([])
# ax5.set_yticks([])
# fig.savefig("figures/encoding.png", bbox_inches="tight", pad_inches=0)

# ''' INPUT RASTER (MNIST) '''
# fig =  plt.figure(facecolor="w", figsize=(10,2))
# ax = gca()
# in_size = dataset_s[0][0].shape[-1]
# dot_size = 0.5
# num_pixels = in_size**2
# round_pixels = int(ceil(num_pixels / 100.0)) * 100
# splt.raster(inputs.reshape(inputs.shape[0], -1), ax, s=dot_size, c="black")

# ax.set(xlim=[0, inputs.shape[0]], ylim=[-50, round_pixels+50], yticks = [], xlabel="Time, ms")
# fig.tight_layout()
# fig.savefig("figures/rasters.png", bbox_inches="tight", pad_inches=0)
# plt.show()

''' INPUT RASTER (DVS) '''
index = 0
sensor_size = tonic.datasets.DVSGesture.sensor_size
new_size = (28, 28, 2)

ds_transform = tonic.transforms.Compose([
            transforms.Denoise(filter_time=1000),
            tonic.transforms.Downsample(spatial_factor=0.21875, time_factor=0.001),
            transforms.ToFrame(sensor_size=new_size, n_time_bins=100),
            torch.from_numpy,
            dtype_transform(),
            SignTransform()
            ])

# create datasets
train_ds = tonic.datasets.DVSGesture("dataset/dvs_gesture", train=True, transform=ds_transform)
train_loader = DataLoader(train_ds,
                          shuffle=True, 
                          batch_size=64, 
                          collate_fn=tonic.collation.PadTensors(batch_first=False))
inputs = train_ds[index][0].squeeze()
fig =  plt.figure(facecolor="w", figsize=(10,2))
ax = gca()
in_size = train_ds[0][0].shape[-1]
dot_size = 0.5
num_pixels = in_size**2
round_pixels = int(ceil(num_pixels / 100.0)) * 100
splt.raster(inputs.reshape(inputs.shape[0], -1), ax, s=dot_size, c="black")

ax.set(xlim=[0, inputs.shape[0]], ylim=[-50, round_pixels+50], yticks = [], xlabel="Time, ms")
fig.tight_layout()

''' RECURRENCE (MNIST) '''
fig = plt.figure(figsize=(10,5))
df = pd.read_csv("recurrency_mnist.csv")
df['recurrence'] = df['recurrence'].map({True: "Recurrence", False: "No Recurrence"})

us, ps = [], []
for i in set(df["kernel_size"]):
    df1 = df[df["kernel_size"] == i]
    rec = df1[df1["recurrence"] == "Recurrence"]
    norec = df1[df1["recurrence"] == "No Recurrence"]
    u, p_value = stats.mannwhitneyu(rec["Test Loss"], norec["Test Loss"])
    us.append(u)
    ps.append(p_value)

ax = sns.boxplot(df, x="kernel_size", y="Test Loss", hue="recurrence", gap=0.1, palette=sns.color_palette("Paired"))
plt.legend(title = None)
plt.ylabel(r'$\text{RMSE}/\bar{X}$')
plt.xlabel("Kernel Size")

ax.set(ylim=(1.2, 3.5))

# Annotate with U and p values
for i, (u, p) in enumerate(zip(us, ps)):
    x = i+0.1
    y = 3
    ax.text(x - 0.25, y, f'U = {u:.2f}', fontsize=9, color='black')
    ax.text(x - 0.25, y - 0.1, f'p = {p:.3f}', fontsize=9, color='black')
    
fig = plt.figure(figsize=(10,5))
ax = sns.histplot(df, x="Test Loss", hue="recurrence", bins=8, multiple='stack')
ax.get_legend().set_title("")
plt.xlabel(r'$\text{RMSE}/\bar{X}$')

    
''' RECURRENCE (DVS) '''
# fig = plt.figure(figsize=(10,5))
# df = pd.read_csv("recurrency_mnist.csv")
# df['recurrence'] = df['recurrence'].map({True: "Recurrence", False: "No Recurrence"})

# us, ps = [], []
# for i in set(df["kernel_size"]):
#     df1 = df[df["kernel_size"] == i]
#     rec = df1[df1["recurrence"] == "Recurrence"]
#     norec = df1[df1["recurrence"] == "No Recurrence"]
#     u, p_value = stats.mannwhitneyu(rec["Test Loss"], norec["Test Loss"])
#     us.append(u)
#     ps.append(p_value)

# ax = sns.boxplot(df, x="kernel_size", y="Test Loss", hue="recurrence", gap=0.1, palette=sns.color_palette("Paired"))
# plt.legend(title = None)
# plt.ylabel(r'$\text{RMSE}/\bar{X}$')
# plt.xlabel("Kernel Size")

# ax.set(ylim=(1.2, 3.5))

# # Annotate with U and p values
# for i, (u, p) in enumerate(zip(us, ps)):
#     x = i+0.1
#     y = 3
#     ax.text(x - 0.25, y, f'U = {u:.2f}', fontsize=9, color='black')
#     ax.text(x - 0.25, y - 0.1, f'p = {p:.3f}', fontsize=9, color='black')
    
# fig = plt.figure(figsize=(10,5))
# ax = sns.histplot(df, x="Test Loss", hue="recurrence", bins=8, multiple='stack')
# ax.get_legend().set_title("")
# plt.xlabel(r'$\text{RMSE}/\bar{X}$')

'''Frequency UMAP Grid'''
# files = ['200_1.csv', '200_50.csv', '200_100.csv', '200_150.csv', '200_200.csv',
#  '150_1.csv', '150_50.csv', '150_100.csv', '150_150.csv', '150_200.csv',
#  '100_1.csv', '100_50.csv', '100_100.csv', '100_150.csv', '100_200.csv',
#  '50_1.csv', '50_50.csv', '50_100.csv', '50_150.csv', '50_200.csv',
#  '1_1.csv', '1_50.csv', '1_100.csv', '1_150.csv', '1_200.csv']
# cmap = mpl.colormaps['viridis']
# s=0.5
# c_range = np.arange(-0.5, 10, 1)
# norm = colors.BoundaryNorm(c_range, cmap.N)

# fig, axes = plt.subplots(5, 5, figsize=(10, 10))
# for i, ax in enumerate(axes.flat):
#     features = pd.read_csv("datafiles/"+files[i])
#     all_labs = features["Labels"]
#     num_labs = len(set(all_labs.to_numpy()))
#     features = features.loc[:, features.columns != 'Labels']
#     tail = os.path.split(files[i])
#     f_name = f"UMAPS/umap_{tail[1]}.png"
#     print(f"{i+1}/{len(axes.flat)}")
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore")
#         umap = UMAP(n_components=2, random_state=42, n_neighbors=15).fit_transform(features)
    
#     f, l = extract_numbers(files[i])
    

#     axi = ax.scatter(umap[:, 0], umap[:, 1], c=all_labs, cmap=cmap, norm=norm, s=s)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     if i >= 20:
#         ax.set_xlabel(f"{l} Hz")
    
#     if i % 5 == 0:
#         ax.set_ylabel(f"{f} Hz")

# # Adjust the subplots to make room for the colorbar
# fig.subplots_adjust(right=1)

# Add a colorbar to the right of the subplots
# cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
# fig.colorbar(axi, ax=cbar_ax, label='Digit Class', ticks=c_range-0.5)

# Add shared x and y labels across the whole figure
# fig.text(0.5, -0.02, r'Minimum Encoding Frequency, $f_\text{Off}$', ha='center', va='center', fontsize=14)
# fig.text(-0.02, 0.5, r'Maximum Encoding Frequency, $f_\text{On}$', ha='center', va='center', rotation='vertical', fontsize=14)

# plt.tight_layout()

'''Node UMAP Grid'''
# files = ['2.csv', '10.csv', '20.csv', '30.csv', '40.csv', '50.csv', '100.csv', '400.csv', '500.csv', '506.csv', '600.csv', '1000.csv']
# cmap = mpl.colormaps['viridis']
# s=0.5
# c_range = np.arange(-0.5, 10, 1)
# norm = colors.BoundaryNorm(c_range, cmap.N)

# fig, axes = plt.subplots(4, 3, figsize=(12, 9))
# for i, ax in enumerate(axes.flat):
#     features = pd.read_csv("datafiles/"+files[i])
#     all_labs = features["Labels"]
#     num_labs = len(set(all_labs.to_numpy()))
#     features = features.loc[:, features.columns != 'Labels']
#     tail = os.path.split(files[i])
#     f_name = f"UMAPS/umap_{tail[1]}.png"
    
#     print(f"{i+1}/{len(axes.flat)}")
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore")
#         umap = UMAP(n_components=2, random_state=42, n_neighbors=15).fit_transform(features)
    
#     #f, l = extract_numbers(files[i])
    

#     axi = ax.scatter(umap[:, 0], umap[:, 1], c=all_labs, cmap=cmap, norm=norm, s=s)
#     ax.set_title(files[i][0:-4])
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     # if i >= 20:
#     #     ax.set_xlabel(f"{l} Hz")
    
#     # if i % 5 == 0:
#     #     ax.set_ylabel(f"{f} Hz")

# # Adjust the subplots to make room for the colorbar
# fig.subplots_adjust(right=1)








'''Frequency Heatmap'''
# labels = ["1", "50", "100", "150", "200"]
# labels = [1, 50, 100, 150, 200]
# hmap = pd.read_csv("datafiles/rates heatmap data.csv", index_col=0)#, header=labels, index_col=labels)

# plt.figure(figsize = (10,7))
# plt.title('Rates Heat Map (Errors normalised by frequency)')
# ax = sns.heatmap(hmap*100, vmin=0.0, vmax=40, cmap = 'cividis', annot = True, fmt = '.2f')
# plt.xlabel('Rate On, Hz')
# plt.ylabel('Rate Off, Hz')
# ax.invert_yaxis()
# ax.set_facecolor('y')
# # plt.savefig('heatmap.png')
# # plt.savefig('heatmap.pdf')


''' Loss v Node '''
plt.figure(figsize = (10,5))
nodes = [1000, 600, 500, 400, 100, 50, 40, 30, 20, 10, 2]
test_loss = [1.285, 1.993, 1.225, 2.453, 1.922, 1.225, 1.348, 1.486, 1.683, 2.097, 2.586]
node_files = sorted(glob("node_cv_s.csv"))
# Create dictionary
df = pd.concat((pd.read_csv(file) for file in node_files), ignore_index=True)
df = df[df["recurrence"]==False]
df["Spiking"] = "SAE"
df_ns = pd.read_csv("node_cv_ns.csv")

ad_hoc = {"num_rec": [445, 446, 447, 448, 449, 450, 451],
          "Test Loss": [0.627, 0.6303, 0.6154, 0.6096, 0.5912, 2.645, 0.6225]}



df_ns = pd.concat([df_ns, pd.DataFrame(ad_hoc)], ignore_index=True)
df_ns["Spiking"] = "AE"


df = pd.concat([df, df_ns], ignore_index=True)

line_plot = sns.lineplot(df, y="Test Loss", x="num_rec", marker='o', hue="Spiking", markersize=3)

# Customize the plot for better visual appeal
line_plot.set_xlabel('Latent Size')
line_plot.set_ylabel(r'$\text{RMSE}/\bar{X}$')
line_plot.tick_params(labelsize=12)
line_plot.grid(True)
#sns.set_style("whitegrid")
plt.legend(title = None)
plt.ylim([0, 1.5])

# NMNIST Input Data

# dataset = tonic.datasets.NMNIST(save_to='./dataset',
#                                       train=True,
#                                       first_saccade_only=False
#                                       )
# targets = set()
# for events, target in dataset:
#     if target not in targets:
#         targets.add(target)
#         tonic.utils.plot_event_grid(events)
#         if len(targets) == 3:
#             break

# sensor_size = tonic.datasets.NMNIST.sensor_size
# print(f"Sensor Size: {sensor_size}")
# raw_transform = tonic.transforms.Compose([
#             transforms.Denoise(filter_time=10000),
#             transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
#             torch.from_numpy,
#             dtype_transform()
#             ])

# dataset = tonic.datasets.NMNIST(save_to='./dataset',
#                                       train=True,
#                                       first_saccade_only=False,
#                                       transform = raw_transform
#                                       )

# print(f"Dataset has {len(dataset)} elements.")
# targets = set()
# im_list = []
# for events, target in dataset:
#     if target not in targets:
#         print(target)
#         print(events.shape)
#         targets.add(target)
#         im_list.append(events)
        
#         ax1 = plt.subplot(1, 3, 1)
#         plt.imshow(im_list[target][1].squeeze())
#         plt.xlabel("t = 1")

#         ax2 = plt.subplot(1, 3, 2)
#         plt.imshow(im_list[target][50].squeeze())
#         plt.xlabel("t = 100")
#         ax3 = plt.subplot(1, 3, 3)
#         plt.imshow(im_list[target][-1].squeeze())
#         plt.xlabel("t = 300")
#         if len(targets) == 3:
#             break



# times = []
# for i, (events, target) in enumerate(dataset):
#     print(f"{i}/{len(dataset)}")    
#     targets.add(target)
#     times.append(events.shape[0])

# print(np.array(times).mean())
# print(np.array(times).std())
plt.show()
