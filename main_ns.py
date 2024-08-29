"""## Imports"""
#print("\nImporting modules and defining functions")

#importing module
from datetime import datetime
date = datetime.now().strftime("%d/%m - %H:%M")
from image_to_image import CAE
import numpy as np
import wandb
from helpers import get_poisson_inputs, build_datasets, build_network,\
    to_np, set_seed, umap_plt, visTensor

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
    set_seed(wandb.config.seed)
    #run.name = "ns_UMAP"
    """## Define network architecutre and parameters"""
    network_params, input_specs, train_specs, convolution_params = {}, {}, {}, {}
    
    # Parameters for use in training
    
    convolution_params["channels_1"] = 12
    convolution_params["filter_1"] = wandb.config.kernel_size
    convolution_params["channels_2"] = 64
    convolution_params["filter_2"] = wandb.config.kernel_size
    
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
    noise = wandb.config.noise
    recurrence = wandb.config.recurrence
    run.name = f"{wandb.config.seed}_{wandb.config.kernel_size}_{wandb.config.num_rec}"
    print(f'Starting Sweep: Batch Size: {train_specs["batch_size"]}, Learning Rate: {train_specs["lr"]}')
    
    # Build dataset and loaders
    train_dataset, train_loader, test_dataset, test_loader = build_datasets(train_specs)
    
    # Build network
    network = CAE(num_rec=num_rec, cp=convolution_params, recurrence=recurrence).to(device)
    visTensor(network)
    # Train network
    network, train_loss, test_loss, final_train_loss, final_test_loss = train_network_ns(network, train_loader, test_loader, train_specs)
    visTensor(network)
    # # Plot examples from MNIST
    # unique_images = []
    # seen_labels = set()
    if train_specs["lr"] == 1e-3 and train_specs["batch_size"] == 64:
        torch.save(network.state_dict(), 'CAE.pth')
    # for image, label in train_dataset:
    #     if label not in seen_labels:
    #         unique_images.append((image, label))
    #         seen_labels.add(label)

    # unique_images.sort(key=lambda x: x[1])

    # fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # axes = axes.flatten()

    # # Loop over each subplot
    # for i, ax in enumerate(axes):
    #     ax.set_title(f'Number: {unique_images[i][1]}')
    #     ax.imshow(unique_images[i][0].reshape(28,28), cmap = 'gray')  # Blank image, you can replace this with your content
    #     ax.axis('off')

    # plt.tight_layout()

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

    # fig, axs = plt.subplots(4, 5, figsize=(12, 10))

    # # Flatten the axis array for easier indexing
    # axs = axs.flatten()

    # # Plot the first 5 images from orig_ims
    # for i in range(5):
    #     axs[i].imshow(orig_ims[i][0], cmap='grey')
    #     print(orig_ims[i][0].shape)
    #     if i==2:
    #         axs[i].set_title('Originals: 0 - 4')
    #     axs[i].axis('off')

    # # Plot the first 5 images from unique_ims
    # for i in range(5):
    #     axs[i+5].imshow(unique_ims[i][0], cmap='grey')
    #     print(unique_ims[i][0].shape)
    #     if i==2:
    #         axs[i+5].set_title('Reconstructions: 0 - 4')
    #     axs[i+5].axis('off')

    # # Plot the remaining images from orig_ims
    # for i in range(5, 10):
    #     axs[i+5].imshow(orig_ims[i][0], cmap='grey')
    #     if i==7:
    #         axs[i+5].set_title('Originals: 5 - 9')
    #     axs[i+5].axis('off')

    # # Plot the remaining images from unique_ims
    # for i in range(5, 10):
    #     axs[i+10].imshow(unique_ims[i][0], cmap='grey')
    #     if i==7:
    #         axs[i+10].set_title('Reconstructions: 5 - 9')
    #     axs[i+10].axis('off')

    # plt.tight_layout()

    # fig.savefig("figures/result_summary.png")
    fig, axs = plt.subplots(1, 10, layout="constrained")
    
    pad = 0
    h = 0
    w = 0
    
    # Flatten the axis array for easier indexing
    axs = axs.flatten()
    
    # Plot the first 5 images from orig_ims
    
    for i in range(10):
        axs[i].imshow(orig_ims[i][0], cmap='grey')
        axs[i].axis('off')
    
    plt.tight_layout(pad=pad, h_pad=h, w_pad=w)
    fig.savefig("figures/mnist_inputs.png", bbox_inches='tight')
    
    fig, axs = plt.subplots(1, 10, layout="constrained")
    
    # Flatten the axis array for easier indexing
    axs = axs.flatten()    
    for i in range(10):
        axs[i].imshow(unique_ims[i][0], cmap='grey')
        axs[i].axis('off')      
    
    plt.tight_layout(pad=pad, h_pad=h, w_pad=w)
   
    fig.savefig(f"figures/ns_result_summary_{run.name}.png", bbox_inches='tight', pad_inches=0)
    
    print("Plotting Spiking Input MNIST")
    # Plot originally input as image and as spiking representation - save gif.
    input_index = 0
    inputs, labels = test_dataset[input_index]
    inputs = inputs.unsqueeze(0)
    img_spk_recs, img_spk_outs = network(inputs)
    inputs = inputs.squeeze().cpu()

    img_spk_outs = img_spk_outs.squeeze().detach().cpu()


    #print("Plotting Spiking Output MNIST")
    #fig, axs = plt.subplots()
    #axs.imshow(img_spk_outs, cmap='grey')

    umap_file, sil_score, db_score, _ = umap_plt("./datafiles/"+run.name+".csv")
    
    wandb.log({"Test Loss": final_test_loss,
                "Results Grid": wandb.Image(f"figures/ns_result_summary_{run.name}.png"),
                "UMAP": wandb.Image(umap_file)
                })
    
    
    del network, train_loss, test_loss, final_train_loss, final_test_loss, \
        train_dataset, train_loader, test_dataset, test_loader
        
    gc.collect()
    torch.cuda.empty_cache()
        
        
if __name__ == '__main__':  
  numrecs = list(range(490, 511, 2)) + list(range(590, 611, 2))
  test = 0
  
  if test == 1:
      sweep_config = {
          'name': f'ns_Test Sweep {date}',
          'method': 'grid',
          'metric': {'name': 'Test Loss',
                      'goal': 'minimize'   
                      },
          'parameters': {'bs': {'values': [64]},
                          'lr': {'values': [1e-4]},
                          'epochs': {'values': [10]},
                          "subset_size": {'values': [10]},
                          "recurrence": {'values': [1]},
                          "noise": {'values': [0]},
                          "rate_on": {'values': [75]},
                          "rate_off": {'values': [1]},
                          "num_workers": {'values': [0]},
                          "num_rec": {'values': [100]},
                          "norm_type": {'values': ["mean"]},
                          "kernel_size": {'values': [7]},
                          "seed": {'values': [42]}
                          }
          }
  else:
      sweep_config = { #REMEMBER TO CHANGE RUN NAME
          'name': f'ns_Kernel Size Sweep {date}', 
          'method': 'grid',
          'metric': {'name': 'Test Loss',
                      'goal': 'minimize'   
                      },
          'parameters': {'bs': {'values': [64]},
                          'lr': {'values': [1e-4]},
                          'epochs': {'values': [10]},
                          "subset_size": {'values': [10]},
                          "recurrence": {'values': [0]},
                          "noise": {'values': [0]},
                          "rate_on": {'values': [75]},
                          "rate_off": {'values': [1]},
                          "num_workers": {'values': [0]},
                          "num_rec": {'values': [100]},
                          "norm_type": {'values': ["mean"]},
                          "kernel_size": {'values': [3, 5, 7, 9, 11, 13]},
                          "seed": {'values': [42]}
                          }
          }
  
  
  sweep_id = wandb.sweep(sweep = sweep_config, project = "MSc Project", entity="lukelalderson")
      
  wandb.agent(sweep_id, function=main_ns)
  
  torch.cuda.empty_cache()