"""## Imports"""
#print("\nImporting modules and defining functions")

#importing module
from datetime import datetime
date = datetime.now().strftime("%d/%m - %H:%M")
from image_to_image import CAE
import numpy as np
import wandb
from helpers import get_poisson_inputs, build_datasets, build_network,\
    to_np, set_seed, umap_plt

import torch
import snntorch.spikeplot as splt
from IPython.display import HTML
from brian2 import *
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from train_network import train_network, train_network_ns

import gc
    
def main_ns():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using CUDA") if torch.cuda.is_available() else print("Using CPU")
    set_seed()
    torch.backends.cudnn.benchmark = True #TURN OFF WHEN CHANGING ARCHITECTURE    

    run = wandb.init()
    run.name = f"{wandb.config.rate_on} Hz_{wandb.config.norm_type} Hz_{wandb.config.noise}"
    """## Define network architecutre and parameters"""
    network_params, input_specs, train_specs, convolution_params = {}, {}, {}, {}
    
    # Parameters for use in training
    
    convolution_params["channels_1"] = 12
    convolution_params["filter_1"] = 3
    convolution_params["channels_2"] = 64
    convolution_params["filter_2"] = 3
    
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
    
    train_specs["norm_type"] = wandb.config.norm_type
    num_rec = wandb.config.num_rec
    
    train_specs["scaler"] = abs(1/((wandb.config.rate_on-wandb.config.rate_off)*(0.001*input_specs["total_time"]/ms))) if wandb.config.rate_on != wandb.config.rate_off else 100000
    
    
    noise = wandb.config.noise
    recurrence = wandb.config.recurrence
    
    print(f'Starting Sweep: Batch Size: {train_specs["batch_size"]}, Learning Rate: {train_specs["lr"]}')
    
    # Build dataset and loaders
    train_dataset, train_loader, test_dataset, test_loader = build_datasets(train_specs)
    
    # Build network
    network = CAE(num_rec=num_rec, cp=convolution_params, recurrence=recurrence).to(device)
    
    # Train network
    network, train_loss, test_loss, final_train_loss, final_test_loss = train_network_ns(network, train_loader, test_loader, train_specs)

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
    
    print("Assembling test data for 2D projection")
    ###
    with torch.no_grad():
        features, all_labs, all_decs, all_orig_ims = [], [], [], []
        for i,(data, labs) in enumerate(test_loader, 1):
            data = data.to(device)
            code_layer, decoded = network(data)
            features.append(to_np(code_layer))
            all_labs.append(labs)
            all_decs.append(decoded.squeeze().cpu())
            all_orig_ims.append(data.squeeze().cpu())
           
            if i % 1 == 0:
                print(f'-- {i}/{len(test_loader)} --')
    
        features = np.concatenate(features, axis = 0) #(N, 100)
        all_labs = np.concatenate(all_labs, axis = 0)
        all_orig_ims = np.concatenate(all_orig_ims, axis = 0)
        all_decs = np.concatenate(all_decs, axis = 0)
  
    tsne = pd.DataFrame(data = features)
    
    tsne.insert(0, "Labels", all_labs) 
    
    tsne.to_csv("./datafiles/"+run.name+".csv", index=False)
    
    
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
    # Plot originally input as image and as spiking representation - save gif.
    input_index = 0
    inputs, labels = test_dataset[input_index]
    inputs = inputs.unsqueeze(0)
    img_spk_recs, img_spk_outs = network(inputs)
    inputs = inputs.squeeze().cpu()
    
    img_spk_outs = img_spk_outs.squeeze().detach().cpu()
    
        
    print("Plotting Spiking Output MNIST")
    fig, axs = plt.subplots()
    axs.imshow(img_spk_outs, cmap='grey')
          
    umap_file, sil_score, db_score = umap_plt("./datafiles/"+run.name+".csv")
    
    wandb.log({"Test Loss": final_test_loss,
                "Results Grid": wandb.Image("figures/result_summary.png"),
                "UMAP": wandb.Image(umap_file)
                })
    
    
    del network, train_loss, test_loss, final_train_loss, final_test_loss, \
        features, all_labs, all_decs, all_orig_ims, \
        train_dataset, train_loader, test_dataset, test_loader, tsne, inputs, \
        img_spk_outs, img_spk_recs, code_layer, decoded
        
    gc.collect()
    torch.cuda.empty_cache()
        
        
if __name__ == '__main__':  
  
  test = 1
  
  if test == 1:
      sweep_config = {
          'name': f'ns_Test Sweep {date}',
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
                          "norm_type": {'values': ["norm"]}
                          }
          }
  else:
      sweep_config = { #REMEMBER TO CHANGE RUN NAME
          'name': f'ns_Assessing different Normalisation - No Scaler {date}',
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
                          "norm_type": {'values': ["norm", "range", "mean", None]} #
                          }
          }
  
  
  sweep_id = wandb.sweep(sweep = sweep_config, project = "MSc Project", entity="lukelalderson")
      
  wandb.agent(sweep_id, function=main_ns)
  
  torch.cuda.empty_cache()