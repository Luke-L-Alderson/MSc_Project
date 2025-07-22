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
import scikit_posthocs as sp
import statsmodels.api as sm

def extract_numbers(file_name):
    base_name = file_name.split('.')[0]  # Remove the extension
    parts = base_name.split('_')  # Split by the underscore
    return int(parts[0]), int(parts[1])


#umap_plt("datafiles/0.01_8.csv", w=10, h=10)

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

''' BOX PLOTS'''

p_adjustment = "holm"
fig_height = 7
fig_width = 7


''' RECURRENCE (MNIST) '''
hue_order = ["Recurrence (RLeaky)", "No Recurrence (1 Layer)", "No Recurrence (2 Layers)"]
ypos = 1
xpos = 0.1
xtextpos = 0.25
ytextpos = 0.03

df = pd.read_csv("mnist_recurrence.csv")
df['recurrence'] = df['recurrence'].map({True: "Recurrence (RLeaky)", False: "No Recurrence (2 Layers)"})
df["Test Loss"] = df["Test Loss"]#/10

df1 = pd.read_csv("mnist_recurrence_ol.csv")
df1['recurrence'] = "No Recurrence (1 Layer)"

df = pd.concat([df, df1], ignore_index=True)

us, ps, mus, stds = [], [], [], []
for i in set(df["kernel_size"]):
    df1 = df[df["kernel_size"] == i]
    rec = df1[df1["recurrence"] == "Recurrence (RLeaky)"]
    norec = df1[df1["recurrence"] == "No Recurrence (2 Layers)"]
    onelayer = df1[df1["recurrence"] == "No Recurrence (1 Layer)"]
    rec_p = stats.shapiro(rec["Test Loss"])
    norec_p = stats.shapiro(norec["Test Loss"])
    onelayer_p = stats.shapiro(onelayer["Test Loss"])
    print(f"Rec Normal? {rec_p.pvalue}, No Rec Normal? {norec_p.pvalue}, onelayer normal? {onelayer_p.pvalue}")
    sm.qqplot(stats.zscore(rec["Test Loss"]), line='45')
    sm.qqplot(stats.zscore(norec["Test Loss"]), line='45')
    sm.qqplot(stats.zscore(onelayer["Test Loss"]), line='45')
    u, p_value = stats.kruskal(rec["Test Loss"], norec["Test Loss"], onelayer["Test Loss"])
    p_dunns = sp.posthoc_dunn([rec["Test Loss"], onelayer["Test Loss"], norec["Test Loss"]], p_adjust=p_adjustment)
    us.append(u)
    ps.append(p_value)
    mus.append(norec["Test Loss"].mean())
    stds.append(norec["Test Loss"].std())
    mus.append(rec["Test Loss"].mean())
    stds.append(rec["Test Loss"].std())
    mus.append(onelayer["Test Loss"].mean())
    stds.append(onelayer["Test Loss"].std())
    print(f"\nFor {i}:\nAll P: {p_value} \nDunns test:\n{p_dunns}")

print(mus)
print("\n")
print(f"{stds}\n")
fig = plt.figure(figsize=(fig_width,fig_height))
ax = sns.boxplot(df, x="kernel_size", y="Test Loss", hue="recurrence", gap=0.1, palette=sns.color_palette("Paired"), hue_order=hue_order)

plt.legend(title = None, fontsize=9)
plt.ylabel("Test Loss")
#plt.ylabel(r'$\text{RMSE}/\bar{X}$')
plt.xlabel("Kernel Size")

ax.set(ylim=(0.4, 1.2))

# Annotate with U and p values
for i, (u, p) in enumerate(zip(us, ps)):
    x = i+0.1
    y = 0.465
    ax.text(x - 0.25, y, f'H = {u:.2f}', fontsize=9, color='black')
    ax.text(x - 0.25, y - 0.03, f'p = {p:.3f}', fontsize=9, color='black')
    
fig = plt.figure(figsize=(fig_width,fig_width))
ax = sns.histplot(df, x="Test Loss", hue="recurrence", bins=12, multiple='stack')
ax.get_legend().set_title("")
plt.ylabel("Test Loss")
    
''' RECURRENCE (DVS) '''

df = pd.read_csv("dvs_recurrence.csv")
df['recurrence'] = df['recurrence'].map({True: "Recurrence (RLeaky)", False: "No Recurrence (2 Layers)"})

df1 = pd.read_csv("dvs_recurrence_ol.csv")
df1['recurrence'] = "No Recurrence (1 Layer)"

df = pd.concat([df, df1], ignore_index=True)

us, ps, mus, stds = [], [], [], []
for i in set(df["kernel_size"]):
    df1 = df[df["kernel_size"] == i]
    rec = df1[df1["recurrence"] == "Recurrence (RLeaky)"]
    norec = df1[df1["recurrence"] == "No Recurrence (2 Layers)"]
    onelayer = df1[df1["recurrence"] == "No Recurrence (1 Layer)"]
    rec_p = stats.shapiro(rec["Test Loss"])
    norec_p = stats.shapiro(norec["Test Loss"])
    onelayer_p = stats.shapiro(onelayer["Test Loss"])
    print(f"Rec Normal? {rec_p.pvalue}, No Rec Normal? {norec_p.pvalue}, onelayer normal? {onelayer_p.pvalue}")
    sm.qqplot(stats.zscore(rec["Test Loss"]), line='45')
    sm.qqplot(stats.zscore(norec["Test Loss"]), line='45')
    sm.qqplot(stats.zscore(onelayer["Test Loss"]), line='45')
    u, p_value = stats.kruskal(rec["Test Loss"], norec["Test Loss"], onelayer["Test Loss"])
    p_dunns = sp.posthoc_dunn([rec["Test Loss"], onelayer["Test Loss"], norec["Test Loss"]], p_adjust=p_adjustment)
    us.append(u)
    ps.append(p_value)
    mus.append(norec["Test Loss"].mean())
    stds.append(norec["Test Loss"].std())
    mus.append(rec["Test Loss"].mean())
    stds.append(rec["Test Loss"].std())
    mus.append(onelayer["Test Loss"].mean())
    stds.append(onelayer["Test Loss"].std())
    print(f"\nFor {i}:\nAll P: {p_value} \nDunns test:\n{p_dunns}")
    
print(mus)
print("\n")
print(stds)
fig = plt.figure(figsize=(fig_width,fig_height))
ax = sns.boxplot(df, x="kernel_size", y="Test Loss", hue="recurrence", gap=0.1, palette=sns.color_palette("Paired"), hue_order=hue_order)

linedif = 0.2669999
x1, x2 = 1-linedif, 1+linedif   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = df[df["kernel_size"] == 5]['Test Loss'].max() + 0.042, 0.02, 'k'

plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

x1, x2 = 1, 1+linedif   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = df[df["kernel_size"] == 5]['Test Loss'].max() + 0.006, 0.02, 'k'

plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col)

x1, x2 = 2, 2+linedif   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = df[df["kernel_size"] == 7]['Test Loss'].max() + 0.006, 0.02, 'k'

plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

plt.legend(title = None, fontsize=9)
plt.ylabel("Test Loss")
#plt.ylabel(r'$\text{RMSE}/\bar{X}$')
plt.xlabel("Kernel Size")

ax.set(ylim=(0.4, 1.2))

# Annotate with U and p values
for i, (u, p) in enumerate(zip(us, ps)):
    x = i+0.1
    y = 1
    ax.text(x - 0.25, y, f'H = {u:.2f}', fontsize=9, color='black')
    ax.text(x - 0.25, y - 0.03, f'p = {p:.3f}', fontsize=9, color='black')


fig = plt.figure(figsize=(fig_width,fig_width))
ax = sns.histplot(df, x="Test Loss", hue="recurrence", bins=12, multiple='stack')
ax.get_legend().set_title("")
plt.ylabel("Test Loss")

''' Recurrence with DVS State '''

df = pd.read_csv("dvs_recurrence_state.csv")
df['recurrence'] = df['recurrence'].map({True: "Recurrence (RLeaky)", False: "No Recurrence (2 Layers)"})

df1 = pd.read_csv("dvs_recurrence_state_ol.csv")
df1['recurrence'] = "No Recurrence (1 Layer)"

df = pd.concat([df, df1], ignore_index=True)
df['state'] = df['state'].map({"present": "Present", "past": "Past", "future":"Future"})

us, ps, mus, stds = [], [], [], []
for i in set(df["state"]):
    df1 = df[df["state"] == i]
    rec = df1[df1["recurrence"] == "Recurrence (RLeaky)"]
    norec = df1[df1["recurrence"] == "No Recurrence (2 Layers)"]
    onelayer = df1[df1["recurrence"] == "No Recurrence (1 Layer)"]
    rec_p = stats.shapiro(rec["Test Loss"])
    norec_p = stats.shapiro(norec["Test Loss"])
    onelayer_p = stats.shapiro(onelayer["Test Loss"])
    print(f"Rec Normal? {rec_p.pvalue}, No Rec Normal? {norec_p.pvalue}, onelayer normal? {onelayer_p.pvalue}")
    sm.qqplot(stats.zscore(rec["Test Loss"]), line='45')
    sm.qqplot(stats.zscore(norec["Test Loss"]), line='45')
    sm.qqplot(stats.zscore(onelayer["Test Loss"]), line='45')
    u, p_value = stats.kruskal(rec["Test Loss"], norec["Test Loss"], onelayer["Test Loss"])
    p_dunns = sp.posthoc_dunn([rec["Test Loss"], onelayer["Test Loss"], norec["Test Loss"]], p_adjust=p_adjustment)
    us.append(u)
    ps.append(p_value)
    mus.append(norec["Test Loss"].mean())
    stds.append(norec["Test Loss"].std())
    mus.append(rec["Test Loss"].mean())
    stds.append(rec["Test Loss"].std())
    mus.append(onelayer["Test Loss"].mean())
    stds.append(onelayer["Test Loss"].std())
    print(f"\nFor {i}:\nAll P: {p_value} \nDunns test:\n{p_dunns}\n")
    
print("\n")
print(mus)
print("\n")
print(stds)
fig = plt.figure(figsize=(fig_width,fig_height))
ax = sns.boxplot(df, x="state", y="Test Loss", hue="recurrence", gap=0.1, palette=sns.color_palette("Paired"), hue_order=hue_order, order=["Past", "Present", "Future"])

# same height or not?
x1, x2 = 0, 0.2669999   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = df[df["state"] == "Past"]['Test Loss'].max() + 0.002, 0.005, 'k'

plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col)

x1, x2 = 1, 1.2669999   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = df[df["state"] == "Present"]['Test Loss'].max() + 0.002, 0.005, 'k'

plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col)

x1, x2 = 2, 2.2669999   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = df[df["state"] == "Future"]['Test Loss'].max() + 0.002, 0.005, 'k'

plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col)

x1, x2 = 2-0.2669999, 2+0.2669999   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = df[df["state"] == "Future"]['Test Loss'].max() + 0.009, 0.005, 'k'

plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

plt.legend(title = None, fontsize=9)
plt.ylabel("Test Loss")
#plt.ylabel(r'$\text{RMSE}/\bar{X}$')
#plt.xticks([0, 1, 2], ["Future", "Past", "Present"]) #["Past", "Future", "Present"]
plt.xlabel("Training Target")

ax.set(ylim=(0.45, 0.6))

# Annotate with U and p values
for i, (u, p) in enumerate(zip(us, ps)):
    print(i, u, p)
    x = i+0.07
    y = 0.47
    ax.text(x - 0.25, y, f'H = {u:.2f}', fontsize=9, color='black')
    ax.text(x - 0.25, y - 0.005, f'p = {p:.5f}', fontsize=9, color='black')
    
fig = plt.figure(figsize=(fig_width,fig_width))
ax = sns.histplot(df, x="Test Loss", hue="recurrence", bins=8, multiple='stack')
ax.get_legend().set_title("")
plt.ylabel("Test Loss")

''' DVS Recurrence in Classification '''

df = pd.read_csv("dvs_classification.csv")
df['recurrence'] = df['recurrence'].map({True: "Recurrence (RLeaky)", False: "No Recurrence (2 Layers)"})

df1 = pd.read_csv("dvs_classification_ol.csv")
df1['recurrence'] = "No Recurrence (1 Layer)"

df = pd.concat([df, df1], ignore_index=True)

df["Accuracy"] = 100 - df["Accuracy"]

us, ps, mus, stds = [], [], [], []
for i in set(df["kernel_size"]):
    df1 = df[df["kernel_size"] == i]
    rec = df1[df1["recurrence"] == "Recurrence (RLeaky)"]
    norec = df1[df1["recurrence"] == "No Recurrence (2 Layers)"]
    onelayer = df1[df1["recurrence"] == "No Recurrence (1 Layer)"]
    rec_p = stats.shapiro(rec["Test Loss"])
    norec_p = stats.shapiro(norec["Test Loss"])
    onelayer_p = stats.shapiro(onelayer["Test Loss"])
    print(f"Rec Normal? {rec_p.pvalue}, No Rec Normal? {norec_p.pvalue}, onelayer normal? {onelayer_p.pvalue}")
    sm.qqplot(stats.zscore(rec["Test Loss"]), line='45')
    sm.qqplot(stats.zscore(norec["Test Loss"]), line='45')
    sm.qqplot(stats.zscore(onelayer["Test Loss"]), line='45')
    u, p_value = stats.kruskal(rec["Accuracy"], norec["Accuracy"], onelayer["Accuracy"])
    p_dunns = sp.posthoc_dunn([rec["Accuracy"], onelayer["Accuracy"], norec["Accuracy"]], p_adjust=p_adjustment)
    us.append(u)
    ps.append(p_value)
    mus.append(norec["Accuracy"].mean())
    stds.append(norec["Accuracy"].std())
    mus.append(rec["Accuracy"].mean())
    stds.append(rec["Accuracy"].std())
    mus.append(onelayer["Accuracy"].mean())
    stds.append(onelayer["Accuracy"].std())
    print(f"\nFor {i}:\nAll P: {p_value} \nDunns test:\n{p_dunns}")
    
print("\n")
print(mus)
print("\n")
print(stds)
fig = plt.figure(figsize=(fig_width,fig_height))
ax = sns.boxplot(df, x="kernel_size", y="Accuracy", hue="recurrence", gap=0.1, palette=sns.color_palette("Paired"), hue_order=hue_order)

# same height or not?
x1, x2 = 2-0.2669999, 2+0.2669999   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
y, h, col = df[df["kernel_size"] == 7]['Accuracy'].max() + 1, 1, 'k'

plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col)

# x1, x2 = 1, 1.2669999   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
# y, h, col = df[df["state"] == "past"]['Accuracy'].max() + 0.002, 0.005, 'k'

# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col)

# x1, x2 = 2, 2.2669999   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
# y, h, col = df[df["state"] == "future"]['Accuracy'].max() + 0.002, 0.005, 'k'

# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col)

# x1, x2 = 2-0.2669999, 2+0.2669999   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
# y, h, col = df[df["state"] == "future"]['Accuracy'].max() + 0.009, 0.005, 'k'

# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

plt.legend(title = None, fontsize=9)
plt.ylabel("Error Rate (%)")
#plt.ylabel(r'$\text{RMSE}/\bar{X}$')
#plt.xticks([0, 1, 2], ["Present", "Past", "Future"])
plt.xlabel("Kernel Size")

ax.set(ylim=(35, 55))

# Annotate with U and p values
for i, (u, p) in enumerate(zip(us, ps)):
    x = i+0.08
    y = 36.5
    if i == 2:
        y = y + 2
        ax.text(x - 0.25, y, f'H = {u:.2f}', fontsize=9, color='black')
        ax.text(x - 0.25, y - 0.75, f'p = {p:.5f}', fontsize=9, color='black')
    else:
        ax.text(x - 0.25, y, f'H = {u:.2f}', fontsize=9, color='black')
        ax.text(x - 0.25, y - 0.75, f'p = {p:.5f}', fontsize=9, color='black')

fig = plt.figure(figsize=(fig_width,fig_width))
ax = sns.histplot(df, x="Accuracy", hue="recurrence", bins=8, multiple='stack')
ax.get_legend().set_title("")
plt.ylabel("Accuracy")

'''Frequency UMAP Grid'''
# files = ['200_1.csv', '200_50.csv', '200_100.csv', '200_150.csv', '200_200.csv',
#   '150_1.csv', '150_50.csv', '150_100.csv', '150_150.csv', '150_200.csv',
#   '100_1.csv', '100_50.csv', '100_100.csv', '100_150.csv', '100_200.csv',
#   '50_1.csv', '50_50.csv', '50_100.csv', '50_150.csv', '50_200.csv',
#   '1_1.csv', '1_50.csv', '1_100.csv', '1_150.csv', '1_200.csv']
# files = [
#     '200_1.csv', '200_50.csv', '200_75.csv', '200_100.csv', '200_150.csv', '200_200.csv',
#     '150_1.csv', '150_50.csv', '150_75.csv', '150_100.csv', '150_150.csv', '150_200.csv',
#     '100_1.csv', '100_50.csv', '100_75.csv', '100_100.csv', '100_150.csv', '100_200.csv',
#     '75_1.csv', '75_50.csv', '75_75.csv', '75_100.csv', '75_150.csv', '75_200.csv',
#     '50_1.csv', '50_50.csv', '50_75.csv', '50_100.csv', '50_150.csv', '50_200.csv',
#     '1_1.csv', '1_50.csv', '1_75.csv', '1_100.csv', '1_150.csv', '1_200.csv'
# ]
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
# plt.figure(figsize = (10,5))
# nodes = [1000, 600, 500, 400, 100, 50, 40, 30, 20, 10, 2]
# test_loss = [1.285, 1.993, 1.225, 2.453, 1.922, 1.225, 1.348, 1.486, 1.683, 2.097, 2.586]
# node_files = sorted(glob("node*.csv"))
# # Create dictionary
# df1 = pd.concat((pd.read_csv(file) for file in node_files), ignore_index=True)
# df1 = df1[df1["recurrence"]==False]

# df1["Test Loss"] = df1["Test Loss"]/2.42
# df1 = pd.concat([df1, pd.read_csv("node_cv_s.csv")], ignore_index=True)
# df1["Spiking"] = "SAE (MNIST)"

# df2 = pd.read_csv("node_dvs.csv")
# df2["Spiking"] = "SAE (DVS)"

# df1_ns = pd.read_csv("node_cv_ns1.csv")
# df2_ns = pd.read_csv("node_cv_ns.csv")
# df1_ns["Test Loss"] = df1_ns["Test Loss"]/2.42
# df1_ns = pd.concat([df1_ns, df2_ns], ignore_index=True)
# df1_ns["Spiking"] = " AE (MNIST)"

# df = pd.concat([df1_ns, df1, df2], ignore_index=True)


# line_plot = sns.lineplot(df, y="Test Loss", x="num_rec", marker='o', hue="Spiking", markersize=3)
# #line_plot = sns.lineplot(df, y="Test Loss", x="num_rec", marker='o', hue="Spiking")

# # Customize the plot for better visual appeal
# line_plot.set_xlabel('Latent Size')
# #line_plot.set_ylabel(r'$\text{RMSE}/\bar{X}$')
# line_plot.set_ylabel("Test Loss")
# line_plot.tick_params(labelsize=12)
# line_plot.grid(True)
# #sns.set_style("whitegrid")
# plt.legend(title = None)
# plt.ylim([0, 1.5])

''' Bad Gradients '''
# # Columns renaming dictionaries
# cols1 = {
#     "100 - Mean Gradients": "100",
#     "500 - Mean Gradients": "500",
#     "550 - Mean Gradients": "550",
#     "600 - Mean Gradients": "600"
# }

# cols2 = {
#     "100 - Testing Loss": "100",
#     "500 - Testing Loss": "500",
#     "550 - Testing Loss": "550",
#     "600 - Testing Loss": "600"
# }

# # Read and process the first dataset
# df1 = pd.read_csv("bad_grads.csv")
# df1 = df1.rename(columns=cols1)
# df1 = df1[["100", "500", "600", "550"]]

# # Read and process the second dataset
# df2 = pd.read_csv("bad_grads_training.csv")
# df2.rename(columns=cols2, inplace=True)
# df2 = df2[["100", "500", "600", "550"]]

# # Plotting the first figure
# fig = plt.figure(figsize=(5, 5))
# ax = sns.lineplot(data=df1)
# ax.set_xlim(df1.index.min(), df1.index.max())  # Set x-axis limits to data range
# ax.set_xticks(range(0, 11))  # Set x-axis ticks from 0 to 10 at every step
# plt.grid(True)
# plt.tight_layout(pad=0, h_pad=0, w_pad=0)
# fig.savefig("figures/bad_grads.png", bbox_inches="tight", pad_inches=0)

# # Plotting the second figure
# fig = plt.figure(figsize=(5, 5))
# ax = sns.lineplot(data=df2)
# ax.set_xlim(df2.index.min(), df2.index.max())  # Set x-axis limits to data range
# ax.set_xticks(range(0, 11))  # Set x-axis ticks from 0 to 10 at every step
# plt.grid(True)
# plt.tight_layout(pad=0, h_pad=0, w_pad=0)
# fig.savefig("figures/bad_grads_training.png", bbox_inches="tight", pad_inches=0)

# ''' Node and Kernel '''
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), layout="constrained", sharex=True)
# df = pd.read_csv("nodesandkernels.csv")
# df["kernel_size"] = df["kernel_size"].astype(int)
# df["Spiking"] = "SAE (MNIST)"

# df1 = pd.read_csv("nsnodesandkernels.csv")
# df2_ns = pd.read_csv("node_cv_ns.csv")
# df2_ns = df2_ns[df2_ns["seed"]==42]
# df1 = pd.concat([df1, df2_ns], ignore_index=True)
# df1["Non-Spiking"] = "AE"

# node_files = sorted(glob("node?.csv"))
# # Create dictionary
# df2 = pd.concat((pd.read_csv(file) for file in node_files), ignore_index=True)
# df2 = df2[df2["recurrence"]==False]
# df2["Test Loss"] = df2["Test Loss"]/2.42

# df = pd.concat([df, df2], ignore_index=True)
# ms = 15
# sns.scatterplot(df, y="Test Loss", x="num_rec", hue="kernel_size", ax=ax1, s=ms)
# sns.scatterplot(df1, y="Test Loss", x="num_rec", hue="kernel_size", ax=ax2, s=ms)
# ax2.set_xlabel('Latent Size')

# # Turn off axes legends
# ax1.legend_.remove()
# ax2.legend_.remove()

# # Add a figure legend
# fig.legend(
#     labels=[3, 5, 7, 9, 11],  # Set the labels for the legend
#     loc='upper center',  # Position the legend
#     ncol=5,  # Set the number of columns
#     frameon=False,  # Remove the frame around the legend,
#     bbox_to_anchor = (0.53, -.01)
# )

# xmini = 400
# xmaxi = 750
# ax1.set_xticks([])
# ax1.set_ylim([0, 1.2])
# ax2.set_ylim([0, 1.2])
# ax1.set_title("SAE (MNIST)")
# ax2.set_title(" AE (MNIST)")
# ax1.axvspan(
#     xmin=xmini,  # Set the start of the shaded region
#     xmax=xmaxi,  # Set the end of the shaded region
#     alpha=0.33,  # Set the transparency of the shaded region
#     color="grey",  # Set the color of the shaded region
#     zorder=0,  # Put shaded region behind plot
# )
# ax2.axvspan(
#     xmin=xmini,  # Set the start of the shaded region
#     xmax=xmaxi,  # Set the end of the shaded region
#     alpha=0.33,  # Set the transparency of the shaded region
#     color="grey",  # Set the color of the shaded region
#     zorder=0,  # Put shaded region behind plot
# )

# ax2.set_xticks(list(range(0, 1001, 100)))
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
