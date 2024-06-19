"""## Imports"""
#print("\nImporting modules and defining functions")

#importing module
from datetime import datetime
date = datetime.now().strftime("%d/%m - %H:%M")

import numpy as np
import wandb
from helpers import get_poisson_inputs, build_datasets, build_network, to_np, set_seed, tsne_plt, pca_plt, umap_plt

import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from torchvision import utils as utls
# from torch.utils.data import Subset

import snntorch.spikeplot as splt

from IPython.display import HTML
from brian2 import *
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from train_network import train_network

import gc
    
def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using CUDA") if torch.cuda.is_available() else print("Using CPU")
    set_seed()
    torch.backends.cudnn.benchmark = True #TURN OFF WHEN CHANGING ARCHITECTURE    

    run = wandb.init()
    run.name = f"{wandb.config.rate_on} Hz_{wandb.config.recurrence} Hz_{wandb.config.noise}"
    """## Define network architecutre and parameters"""
    network_params, input_specs, train_specs = {}, {}, {}
    
    # Parameters for use in training
    input_specs["rate_on"] = wandb.config.rate_on*Hz
    input_specs["rate_off"] = wandb.config.rate_off*Hz    
    input_specs["total_time"] = 200*ms
    input_specs["bin_size"] = 1*ms
    
    train_specs["code"] = 'rate'
    
    train_specs["early_stop"] = -1
    train_specs["loss_fn"] = "spike_count"
    train_specs["lambda_rate"] = 0.0
    train_specs["lambda_weights"] = None
    train_specs["num_epochs"] = wandb.config.epochs    
    train_specs["device"] = device
    train_specs["lr"] = wandb.config.lr    
    train_specs["batch_size"] = wandb.config.bs
    train_specs["subset_size"] = wandb.config.subset_size
    train_specs["num_workers"] = wandb.config.num_workers
    
    
    train_specs["scaler"] = abs(1/((wandb.config.rate_on-wandb.config.rate_off)*(0.001*input_specs["total_time"]/ms))) if wandb.config.rate_on != wandb.config.rate_off else 100000
    
    
    noise = wandb.config.noise
    recurrence = wandb.config.recurrence
    
    print(f'Starting Sweep: Batch Size: {train_specs["batch_size"]}, Learning Rate: {train_specs["lr"]}')
    
    # build dataset and loaders
    train_dataset, train_loader, test_dataset, test_loader = build_datasets(train_specs)
    
    # build network
    network, network_params = build_network(device, noise=noise, recurrence=recurrence)
    
    # train network
    network, train_loss, test_loss, final_train_loss, final_test_loss = train_network(network, train_loader, test_loader, input_specs, train_specs)

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
       
    # Plot originally input as image and as spiking representation - save gif.
    inputs, labels = next(iter(test_loader))
    poisson_inputs = get_poisson_inputs(inputs, **input_specs)
    img_spk_recs, img_spk_outs = network(poisson_inputs)
    
    print(poisson_inputs.shape)
    print(img_spk_outs.shape)
    print("Assembling test data for t-sne projection")
    ###
    with torch.no_grad():
       features, all_labs, all_decs, all_orig_ims = [], [], [], []
       for i,(data, labs) in enumerate(test_loader, 1):
           data = get_poisson_inputs(data, **input_specs)
           #print(input_specs)
           code_layer, decoded = network(data)
           code_layer = code_layer.mean(0)
           features.append(to_np(code_layer))#.view(-1, code_layer.shape[1])))
           all_labs.append(labs)
           all_decs.append(decoded.mean(0).squeeze().cpu())
           all_orig_ims.append(data.mean(0).squeeze())
           
           if i % 1 == 0:
               print(f'-- {i}/{len(test_loader)} --')
    
       features = np.concatenate(features, axis = 0) #(N, 100)
       all_labs = np.concatenate(all_labs, axis = 0)
       all_orig_ims = np.concatenate(all_orig_ims, axis = 0)
       all_decs = np.concatenate(all_decs, axis = 0)
  
    tsne = pd.DataFrame(data = features)
    
    tsne.insert(0, "Labels", all_labs) 
    tsne.to_csv(f"{run.name}_{date}")
    
    
    print("Plotting Results Grid")
    seen_labels = set()
    unique_ims = []
    orig_ims = []
    for i, label in enumerate(all_labs):
        if label not in seen_labels:
            seen_labels.add(label)
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
    
    fig.savefig("figures/result_summary.png")
    
    print("Plotting Spiking Input MNIST")
    # img
    input_index = 0
    poisson_inputs = poisson_inputs.squeeze().cpu()
    img_spk_outs = img_spk_outs.squeeze().detach().cpu()
    
    fig, ax = plt.subplots()
    plt.imshow(to_np(inputs[input_index, 0]), cmap = 'grey')
    
    fig, ax = plt.subplots()
    plt.imshow(poisson_inputs[:, input_index].mean(axis=0), cmap='grey')
    
    print(f"Plotting Spiking Input MNIST Animation - {labels[input_index]}")
    fig, ax = plt.subplots()
    anim = splt.animator(poisson_inputs[:, input_index], fig, ax)
    HTML(anim.to_html5_video())
    anim.save(f"figures/spike_mnist_{labels[input_index]}.gif")
    
    wandb.log({"Spike Animation": wandb.Video("spike_mnist.gif", fps=4, format="gif")}, commit = False)
    
    print("Plotting Spiking Output MNIST")
    fig, axs = plt.subplots()
    axs.imshow(img_spk_outs[:, input_index].mean(axis=0), cmap='grey')
    
    print(f"Plotting Spiking Output MNIST Animation - {labels[input_index]}")
    fig1, ax1 = plt.subplots()
    animrec = splt.animator(img_spk_outs[:, input_index], fig1, ax1)
    HTML(animrec.to_html5_video())
    animrec.save(f"figures/spike_mnistrec_{labels[input_index]}.gif")
    
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(poisson_inputs[:, input_index].reshape(200, -1), ax, s=1.5, c="black")
    fig.savefig("figures/input_raster.png")
    
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    splt.raster(img_spk_outs[:, input_index].reshape(200, -1), ax, s=1.5, c="black")
    ax.set_xlim([0, 200])
    
    fig.savefig("figures/output_raster.png")
    
    # if not os.path.exists(newpath):
    #     os.makedirs(newpath)
        
    umap_file = umap_plt(run.name)    
    #tsne_file = tsne_plt(run.name)

    #pca_file = pca_plt(run.name)
    #sns.scatterplot(tsne, x="Feat" "" hue="label")
    
    wandb.log({"Test Loss": final_test_loss,
               "Results Grid": wandb.Image("figures/result_summary.png"),
               #"t-SNE": wandb.Table(tsne),
               #"t-SNE": wandb.Image(tsne_file),
               "UMAP": wandb.Image(umap_file),
               #"PCA": wandb.Image(pca_file),
               "Spike Animation": wandb.Video(f"figures/spike_mnistrec_{labels[input_index]}.gif", fps=4, format="gif"),
               "Input Raster": wandb.Image("figures/input_raster.png"),
               "Output Raster": wandb.Image("figures/output_raster.png")})
    
    
    
    del network, train_loss, test_loss, final_train_loss, final_test_loss, \
        features, all_labs, all_decs, all_orig_ims, \
        train_dataset, train_loader, test_dataset, test_loader, tsne
        
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
                          'lr': {'values': [1e-4]},
                          'epochs': {'values': [1]},
                          "subset_size": {'values': [100]},
                          "recurrence": {'values': [1]},
                          "noise": {'values': [0]},
                          "rate_on": {'values': [1]},
                          "rate_off": {'values': [1]},
                          "num_workers": {'values': [0]}
                          }
          }
  else:
      sweep_config = { #REMEMBER TO CHANGE RUN NAME
          'name': f'Impact of Noise (6-7) {date}',
          'method': 'grid',
          'metric': {'name': 'Test Loss',
                      'goal': 'minimize'   
                      },
          'parameters': {'bs': {'values': [64]},
                          'lr': {'values': [1e-4]},
                          'epochs': {'values': [9]},
                          "subset_size": {'values': [10]},
                          "recurrence": {'values': [1]}, #1, 0.1, 0.5, 0, 1.25, 1.5, 1.75, 2
                          "noise": {'values': [0]},
                          "rate_on": {'values': [75]}, # 25, 50, 75, 100, 125
                          "rate_off": {'values': [1]},
                          "num_workers": {'values': [0]} # ADD REC NEURON PARAM
                          }
          }
  
  
  sweep_id = wandb.sweep(sweep = sweep_config, project = "MSc Project", entity="lukelalderson")
      
  wandb.agent(sweep_id, function=main)
  
  torch.cuda.empty_cache()