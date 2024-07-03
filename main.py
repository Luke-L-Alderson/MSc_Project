"""## Imports"""
#print("\nImporting modules and defining functions")

#importing module
from datetime import datetime
date = datetime.now().strftime("%d/%m - %H:%M")

import numpy as np
import wandb
from helpers import get_poisson_inputs, build_datasets, build_network,\
    to_np, set_seed, umap_plt, weight_map, build_nmnist_dataset

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import snntorch.spikeplot as splt
from IPython.display import HTML
from brian2 import *
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from train_network import train_network

import gc
    
def main():
    run = wandb.init()
    N_MNIST = True if wandb.config.dataset == "NMNIST" else False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using CUDA") if torch.cuda.is_available() else print("Using CPU")
    set_seed()
    torch.backends.cudnn.benchmark = True if not N_MNIST else False #TURN OFF WHEN CHANGING ARCHITECTURE    

    
    run.name = f"{wandb.config.dataset}_{wandb.config.rate_on}_{wandb.config.recurrence}_{wandb.config.noise}"
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
    train_specs["batch_size"] = wandb.config.bs# if not N_MNIST else 16
    train_specs["subset_size"] = wandb.config.subset_size
    train_specs["num_workers"] = wandb.config.num_workers
    train_specs["norm_type"] = wandb.config.norm_type
    num_rec = wandb.config.num_rec
    learnable = wandb.config.learnable
    noise = wandb.config.noise
    recurrence = wandb.config.recurrence
    
    print(f'Starting Sweep: Batch Size: {train_specs["batch_size"]}, Learning Rate: {train_specs["lr"]}')
    
    # Build dataset and loaders
    train_dataset, train_loader, test_dataset, test_loader = build_datasets(train_specs, input_specs)# if not N_MNIST else build_nmnist_dataset(train_specs)
    
    img_size = train_dataset[0][0].shape[-1]
    
    # Build network
    network, network_params = build_network(device, noise=noise, recurrence=recurrence, num_rec=num_rec, learnable=learnable, size=img_size)
    
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
            data = data.transpose(0,1)# if data.shape[0] < data.shape[1] else data 
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
    input_index = 0
    inputs, labels = test_dataset[input_index]
    img_spk_recs, img_spk_outs = network(inputs)
    inputs = inputs.squeeze().cpu()
    img_spk_outs = img_spk_outs.squeeze().detach().cpu()
    
    print(f"Plotting Spiking Input MNIST Animation - {labels}")
    fig, ax = plt.subplots()
    anim = splt.animator(inputs, fig, ax)
    HTML(anim.to_html5_video())
    anim.save(f"figures/spike_mnist_{labels}.gif")


    num_pixels = inputs.shape[1]*inputs.shape[2]
    round_pixels = int(ceil(num_pixels / 100.0)) * 100
    wandb.log({"Spike Animation": wandb.Video(f"figures/spike_mnist_{labels}.gif", fps=4, format="gif")}, commit = False)

    print("Plotting Spiking Output MNIST")
    fig, axs = plt.subplots()
    axs.imshow(img_spk_outs.mean(axis=0), cmap='grey')

    print(f"Plotting Spiking Output MNIST Animation - {labels}")
    fig1, ax1 = plt.subplots()
    animrec = splt.animator(img_spk_outs, fig1, ax1)
    HTML(animrec.to_html5_video())
    animrec.save(f"figures/spike_mnistrec_{labels}.gif")
      
    print("Rasters")
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
        
        fig.savefig(f"figures/weightmap_{run.name}.png")
        plt.show()   
    
    wandb.log({"Test Loss": final_test_loss,
                "Results Grid": wandb.Image("figures/result_summary.png"),
                "UMAP": wandb.Image(umap_file),
                "Spike Animation": wandb.Video(f"figures/spike_mnistrec_{labels}.gif", fps=4, format="gif"),
                "Raster": wandb.Image("figures/rasters.png"),
                "Sil Score": sil_score,
                "DB Score": db_score,
                })
    
    try:
        wandb.log({"Weights": wandb.Image("figures/weightmap.png")})
    except:
        pass
    
    del network, train_loss, test_loss, final_train_loss, final_test_loss, \
        features, all_labs, all_decs, all_orig_ims, \
        train_dataset, train_loader, test_dataset, test_loader, tsne, inputs, \
        img_spk_outs, img_spk_recs, code_layer, decoded, fig, ax
        
    gc.collect()
    torch.cuda.empty_cache()
        
        
if __name__ == '__main__':  
  
  test = 1
  
  
  if test == 1:
      sweep_config = {
          'name': f'Test Sweep {date}',
          'method': 'grid',
          'metric': {'name': 'Test Loss',
                      'goal': 'minimize'   
                      },
          'parameters': {'bs': {'values': [64]},
                          'lr': {'values': [1e-4]},
                          'epochs': {'values': [9]},
                          "subset_size": {'values': [10]},
                          "recurrence": {'values': [1]},
                          "noise": {'values': [0]},
                          "rate_on": {'values': [75]},
                          "rate_off": {'values': [1]},
                          "num_workers": {'values': [0]},
                          "num_rec": {'values': [100]},
                          "norm_type": {'values': ["norm"]},
                          "learnable": {'values': [True]},
                          "dataset": {'values': ["MNIST"]}
                          }
          }
  else:
      sweep_config = { #REMEMBER TO CHANGE RUN NAME
          'name': f'Recurrency and Noise (4-10) {date}',
          'method': 'grid',
          'metric': {'name': 'Test Loss',
                      'goal': 'minimize'   
                      },
          'parameters': {'bs': {'values': [64]},
                          'lr': {'values': [1e-4]},
                          'epochs': {'values': [9]},
                          "subset_size": {'values': [10]},
                          "recurrence": {'values': [0, 1]}, #1, 0.1, 0.5, 0, 1.25, 1.5, 1.75, 2
                          "noise": {'values': [4, 5, 6, 6.2, 6.4, 6.6, 6.8, 7, 8, 9, 10]},
                          "rate_on": {'values': [75]},
                          "rate_off": {'values': [1]},
                          "num_workers": {'values': [0]},
                          "num_rec": {'values': [100]},
                          "norm_type": {'values': ["norm"]},
                          "learnable": {'values': [True]},
                          "dataset": {'values': ["MNIST", "NMNIST"]} #
                          }
          }
  
  
  sweep_id = wandb.sweep(sweep = sweep_config, project = "MSc Project", entity="lukelalderson")
      
  wandb.agent(sweep_id, function=main)
  
  torch.cuda.empty_cache()