"""## Imports"""
print("\nImporting modules and defining functions")

#importing module
import sys
sys.path.append('snn-project')
from datetime import datetime
date = datetime.now().strftime("%d/%m - %H:%M")
import random as rand
import numpy as np
import wandb
from helpers import *
from sklearn.manifold import TSNE


import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import utils as utls
from torch.utils.data import Subset

import snntorch as snn
import snntorch.spikeplot as splt
from snntorch import utils
from snntorch import surrogate

from IPython.display import HTML
from brian2 import *
from brian2.devices import reinit_devices

from model.aux.functions import get_poisson_inputs
from model.train.train_network import train_network

import gc

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Using CUDA") if torch.cuda.is_available() else print("Using CPU")

set_seed()
torch.backends.cudnn.benchmark = True #TURN OFF WHEN CHANGING ARCHITECTURE    
    
def main():

    run = wandb.init()
    run.name = f"{wandb.config.lr}_{wandb.config.bs}_{wandb.config.subset_size}"
    
    """## Define network architecutre and parameters"""
    # get MNIST in, get correct targets, try and vary some biophys params with plots
    time_params, network_params, oscillation_params, frame_params, \
    convolution_params, input_specs, label_specs, train_specs = {}, {}, {}, {}, {}, {}, {}, {}
    
    
    # Parameters for use in training
    input_specs["total_time"] = 200*ms
    input_specs["bin_size"] = 1*ms
    input_specs["rate_on"] = 75*Hz
    input_specs["rate_off"] = 10*Hz
    
    label_specs["total_time"] = 200*ms
    label_specs["code"] = 'rate'
    label_specs["rate"] = 75*Hz
    
    train_specs["num_epochs"] = wandb.config.epochs#3
    train_specs["early_stop"] = -1
    train_specs["device"] = device
    train_specs["lr"] = wandb.config.lr #1e-4
    train_specs["loss_fn"] = "spike_count"
    train_specs["lambda_rate"] = 0.0
    train_specs["lambda_weights"] = None
    train_specs["batch_size"] = wandb.config.bs #64
    train_specs["subset_size"] = wandb.config.subset_size
    
    print(f'Starting Sweep: Batch Size: {train_specs["batch_size"]}, Learning Rate: {train_specs["lr"]}')
    
    #build dataset and loaders
    train_dataset, train_loader, test_dataset, test_loader = build_datasets(train_specs["batch_size"], train_specs["subset_size"])
    
    #build network
    network, network_params = build_network(device)
    

    """## Training the network"""
    network, train_loss, test_loss, final_train_loss, final_test_loss = train_network(network, train_loader, test_loader, input_specs, label_specs, train_specs)
    
    

    # Plot examples from MNIST
    unique_images = []
    seen_labels = set()
    
    for image, label in train_dataset:
        if label not in seen_labels:
            unique_images.append((image, label))
            seen_labels.add(label)
    
    unique_images.sort(key=lambda x: x[1])
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    axes = axes.flatten()
    
    # Loop over each subplot
    for i, ax in enumerate(axes):
        ax.set_title(f'Number: {unique_images[i][1]}')
        ax.imshow(unique_images[i][0].reshape(28,28), cmap = 'gray')  # Blank image, you can replace this with your content
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    input_specs["rate_on"] = 500*Hz
    input_specs["rate_off"] = 10*Hz
    
    # Plot originally input as image and as spiking representation - save gif.
    inputs, labels = next(iter(test_loader))
    poisson_inputs = get_poisson_inputs(inputs, **input_specs)
    
    img_spk_recs, img_spk_outs = network(poisson_inputs)
    
    print("Assembling test data for t-sne projection")
    ###
    with torch.no_grad():
       features, all_labs, all_decs, all_orig_ims = [], [], [], []
       for i,(data, labs) in enumerate(test_loader):
           data = get_poisson_inputs(data, **input_specs)
           code_layer, decoded = network(data)
           code_layer = code_layer.mean(0)
           print(f'-- {i+1}/{len(test_loader)} --')
           features.append(to_np(code_layer.view(-1, code_layer.shape[1])))
           all_labs.append(labs)
           all_decs.append(decoded.mean(0).squeeze().cpu())
           all_orig_ims.append(data.mean(0).squeeze())
    
       features = np.concatenate(features, axis = 0)
       #print(features)
       all_labs = np.concatenate(all_labs, axis = 0)
       all_orig_ims = np.concatenate(all_orig_ims, axis = 0)
       all_decs = np.concatenate(all_decs, axis = 0)
  
    
    try:
        print("Applying t-SNE")
        # Apply TSNE for dimensionality reduction
        tsne = TSNE().fit_transform(features)
        
        # Plot the TSNE results with label
        plt.figure(figsize=(10, 6))
        plt.scatter(tsne[:, 0], tsne[:, 1], c=all_labs, cmap='viridis')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar(label='Digit Class')
        plt.savefig("tsne.png")
        plt.show()
    except Exception:
        print("t-SNE Experienced an Error")
        pass
    
    print("Plotting Results Grid")
    seen_labels = set()
    unique_ims = []
    orig_ims = []
    for i, label in enumerate(all_labs):
        if label not in seen_labels:
            seen_labels.add(label)
            #print(f'{label} added')
            unique_ims.append((all_decs[i], label))
            orig_ims.append((all_orig_ims[i], label))
    
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
    plt.show()
    fig.savefig("result_summary.png")
    
    print("Plotting Spiking Input MNIST")
    # img
    input_index = 0
    poisson_inputs = poisson_inputs.squeeze().cpu()
    img_spk_outs = img_spk_outs.squeeze().detach().cpu()
    
    plt.imshow(to_np(inputs[input_index, 0]), cmap = 'grey')
    plt.show()
    
    plt.imshow(poisson_inputs[:, input_index].mean(axis=0), cmap='grey')
    
    print("Plotting Spiking Input MNIST Animation")
    fig, ax = plt.subplots()
    anim = splt.animator(poisson_inputs[:, input_index], fig, ax)
    HTML(anim.to_html5_video())
    anim.save("spike_mnist.gif")
    plt.show()
    
    wandb.log({"Spike Animation": wandb.Video("spike_mnist.gif", fps=4, format="gif")}, commit = False)
    
    '''
    # Training Loss
    fig, axs = plt.subplots()
    axs.plot(np.array(train_loss), 'k')
    axs.grid(True)
    axs.set_ylabel("MSE of Spike Count")
    axs.set_xlabel("Epochs")
    wandb.log({"Training Loss Plot": wandb.Image(fig)}, commit = False)
    fig.savefig("train_loss.png")
    '''
    
    print("Plotting Spiking Output MNIST")
    fig, axs = plt.subplots()
    axs.imshow(img_spk_outs[:, input_index].mean(axis=0), cmap='grey')
    
    print("Plotting Spiking Output MNIST Animation")
    fig1, ax1 = plt.subplots()
    animrec = splt.animator(img_spk_outs[:, input_index], fig1, ax1)
    HTML(animrec.to_html5_video())
    animrec.save("spike_mnistrec.gif")
    plt.show()
    
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(poisson_inputs[:, input_index].reshape(200, -1), ax, s=1.5, c="black")
    fig.savefig("input_raster.png")
    
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(img_spk_outs[:, input_index].reshape(200, -1), ax, s=1.5, c="black")
    ax.set_xlim([0, 200])
    
    fig.savefig("output_raster.png")
    
    wandb.log({"Test Loss": final_test_loss,
               "Results Grid": wandb.Image("result_summary.png"),
               "t-SNE": wandb.Image("tsne.png"),
               "Spike Animation": wandb.Video("spike_mnistrec.gif", fps=4, format="gif"),
               "Input Raster": wandb.Image("input_raster.png"),
               "Output Raster": wandb.Image("output_raster.png")})
    
    del network, train_loss, test_loss, final_train_loss, \
        features, all_labs, all_decs, all_orig_ims, \
        train_dataset, train_loader, test_dataset, test_loader
        
    gc.collect()
    torch.cuda.empty_cache()
    
    
    
if __name__ == '__main__':
    
    test = 0
    
    if test == 1:
        sweep_config = {
            'name': f'Test Sweep {date}',
            'method': 'grid',
            'metric': {'name': 'Test Loss',
                        'goal': 'minimize'   
                        },
            'parameters': {'bs': {'values': [64]},
                            'lr': {'values': [1e-2]},
                            'epochs': {'values': [1]},
                            "subset_size": {'values': [0.1]}
                            }
            }
    else:
        sweep_config = {
            'name': f'Base Performance Tuning {date}',
            'method': 'bayes',
            'metric': {'name': 'Test Loss',
                        'goal': 'minimize'   
                        },
            'parameters': {'bs': {'values': [64, 32]},
                            'lr': {'values': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]},
                            'epochs': {'values': [3, 6, 9]},
                            "subset_size": {'values': [0.1, 0.05]}
                            }
            }
    
    
    sweep_id = wandb.sweep(sweep = sweep_config, project = "MSc Project")
        
    wandb.agent(sweep_id, function=main, count = 8)