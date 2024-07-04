import tonic
import torch
import tonic.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from tonic import DiskCachedDataset
from torchvision.transforms import v2
from matplotlib import pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML
from helpers import build_network, set_seed, to_np, umap_plt, weight_map, dtype_transform, build_nmnist_dataset
from train_network import train_network
import snntorch as snn
import random
import seaborn as sns
import pandas as pd
from math import ceil

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using CUDA") if torch.cuda.is_available() else print("Using CPU")
set_seed()



train_specs = {}
train_specs["batch_size"] = 64


train_specs["code"] = 'rate'

train_specs["early_stop"] = -1
train_specs["loss_fn"] = "spike_count"
train_specs["lambda_rate"] = 0.0
train_specs["lambda_weights"] = None
train_specs["num_epochs"] = 9  
train_specs["device"] = device
train_specs["lr"] = 1e-4 
train_specs["subset_size"] = 10
train_specs["num_workers"] = 0
train_specs["norm_type"] = "norm"



# Build network
noise = 0
recurrence = 1
num_rec = 100
learnable = True


train_dataset, train_loader, test_dataset, test_loader = build_nmnist_dataset(train_specs)
network, network_params = build_network(device,
                                        noise=noise, 
                                        recurrence=recurrence, 
                                        num_rec=num_rec, 
                                        learnable=learnable, 
                                        depth=1, 
                                        size=train_dataset[0][0].shape[-1])

print(data_shape:=train_dataset[0][0].shape[-1])
# Train network
network, train_loss, test_loss, final_train_loss, final_test_loss = train_network(network, train_loader, test_loader, train_specs)

# Plot examples from MNIST
unique_images = []
seen_labels = set()

for image, label in train_dataset:
    if label not in seen_labels:
        print(label)
        unique_images.append((image.mean(0), label))
        seen_labels.add(label)
        if len(seen_labels) == 10:
            break

unique_images.sort(key=lambda x: x[1])

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

axes = axes.flatten()

# Loop over each subplot
for i, ax in enumerate(axes):
    ax.set_title(f'Number: {unique_images[i][1]}')
    ax.imshow(unique_images[i][0].squeeze(), cmap = 'gray')  # Blank image, you can replace this with your content
    ax.axis('off')

plt.tight_layout()

print("Assembling test data for 2D projection")
###
with torch.no_grad():
    features, all_labs, all_decs, all_orig_ims = [], [], [], []
    for i,(data, labs) in enumerate(test_loader, 1):
        data = data
        code_layer, decoded = network(data)
        code_layer = code_layer.mean(0)
        features.append(to_np(code_layer))
        all_labs.append(labs)
        all_decs.append(decoded.mean(0).squeeze().cpu())
        all_orig_ims.append(data.mean(0).squeeze())
       
        if i % 1 == 0:
            print(f'-- {i}/{len(test_loader)} --')

    features = np.concatenate(features, axis = 0)
    all_labs = np.concatenate(all_labs, axis = 0)
    all_orig_ims = np.concatenate(all_orig_ims, axis = 0)
    all_decs = np.concatenate(all_decs, axis = 0)
    print(type(features))
    print(type(all_labs))
    print((features))
    print((all_labs))

tsne = pd.DataFrame(data = features)
tsne.insert(0, "Labels", all_labs) 
tsne.to_csv("test.csv", index=False)


print("Plotting Results Grid")
seen_labels = set()
unique_ims = []
orig_ims = []
for i, label in enumerate(all_labs):
    if label not in seen_labels:
        seen_labels.add(label)
        unique_ims.append((all_decs[i], label))
        orig_ims.append((all_orig_ims[i], label))
        if len(seen_labels) == 10:
            break

unique_ims.sort(key=lambda x: x[1])
orig_ims.sort(key=lambda x: x[1])

fig, axs = plt.subplots(4, 5, figsize=(12, 10))

# Flatten the axis array for easier indexing
axs = axs.flatten()

# Plot the first 5 images from orig_ims
for i in range(5):
    axs[i].imshow(orig_ims[i][0], cmap='grey')
    if i==2:
        axs[i].set_title('Originals: 0 - 4')
    axs[i].axis('off')

# Plot the first 5 images from unique_ims
for i in range(5):
    axs[i+5].imshow(unique_ims[i][0], cmap='grey')
    if i==2:
        axs[i+5].set_title('Reconstructions: 0 - 4')
    axs[i+5].axis('off')

# Plot the remaining images from orig_ims
for i in range(5, 10):
    axs[i+5].imshow(orig_ims[i][0], cmap='grey')
    if i==7:
        axs[i+5].set_title('Originals: 5 - 9')
    axs[i+5].axis('off')

# Plot the remaining images from unique_ims
for i in range(5, 10):
    axs[i+10].imshow(unique_ims[i][0], cmap='grey')
    if i==7:
        axs[i+10].set_title('Reconstructions: 5 - 9')
    axs[i+10].axis('off')

plt.tight_layout()

fig.savefig("figures/result_summary.png")

print("Plotting Spiking Input MNIST")
# Plot originally input as image and as spiking representation - save gif.
input_index = 0
inputs, labels = test_dataset[input_index]
print(f"Plotting Spiking Input MNIST Animation - {labels}")
fig, ax = plt.subplots()
anim = splt.animator(torch.tensor(inputs[:, 0, :, :]), fig, ax)
HTML(anim.to_html5_video())
anim.save(f"figures/spike_mnist_{labels}.gif")

inputs = torch.tensor(inputs).unsqueeze(1)


img_spk_recs, img_spk_outs = network(inputs)
inputs = inputs.squeeze().cpu()
print(img_spk_outs.shape)
img_spk_outs = img_spk_outs.squeeze().detach().cpu()
print(type(img_spk_outs))
print(img_spk_outs.shape)


#wandb.log({"Spike Animation": wandb.Video(f"figures/spike_mnist_{labels}.gif", fps=4, format="gif")}, commit = False)

print("Plotting Spiking Output MNIST")
fig, axs = plt.subplots()
axs.imshow(img_spk_outs.mean(axis=0), cmap='grey')

print(f"Plotting Spiking Output MNIST Animation - {labels}")
fig1, ax1 = plt.subplots()
animrec = splt.animator(img_spk_outs, fig1, ax1)
HTML(animrec.to_html5_video())
animrec.save(f"figures/spike_mnistrec_{labels}.gif")
  
print("Rasters")
num_pixels = inputs.shape[1]*inputs.shape[2]
round_pixels = int(ceil(num_pixels / 100.0)) * 100
print(round_pixels)
fig = plt.figure(facecolor="w", figsize=(10, 10))
ax1 = plt.subplot(3, 1, 1)
splt.raster(inputs.reshape(inputs.shape[0], -1), ax1, s=1.5, c="black")
ax2 = plt.subplot(3, 1, 2)
splt.raster(img_spk_recs.reshape(inputs.shape[0], -1), ax2, s=1.5, c="black")
ax3 = plt.subplot(3, 1, 3)
splt.raster(img_spk_outs.reshape(inputs.shape[0], -1), ax3, s=1.5, c="black")

ax1.set(xlim=[0, inputs.shape[0]], ylim=[-50, round_pixels+50], xticks=[], ylabel="Neuron Index")
ax2.set(xlim=[0, inputs.shape[0]], ylim=[0-round(num_rec*0.1), round(num_rec*1.1)], xticks=[], ylabel="Neuron Index")
ax3.set(xlim=[0, inputs.shape[0]], ylim=[-50, round_pixels+50], ylabel="Neuron Index", xlabel="Time, ms")
fig.tight_layout()
fig.savefig("figures/rasters.png") 

umap_file, sil_score, db_score = umap_plt("test.csv")

if recurrence == 1:
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax1 = plt.subplot(1,2,1)
    weight_map(network.rlif_rec.recurrent.weight)
    ax2 = plt.subplot(1,2,2)
    sns.histplot(to_np(torch.flatten(network.rlif_rec.recurrent.weight)))
    
    #fig.savefig(f"figures/weightmap_{run.name}.png")
    plt.show() 