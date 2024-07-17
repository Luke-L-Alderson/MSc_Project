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

    # Plotting
    labels, umap_file, sil_score, db_score = plotting_data(network, train_dataset, test_dataset, train_loader, test_loader, recurrence, device, run)
    
    wandb.log({"Test Loss": final_test_loss,
                "Results Grid": wandb.Image("figures/result_summary.png"),
                "UMAP": wandb.Image(umap_file)
                })
    
    
    del network, train_loss, test_loss, final_train_loss, final_test_loss, \
        train_dataset, train_loader, test_dataset, test_loader
        
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
                          "num_rec": {'values': [500, 1000]},
                          "norm_type": {'values': ["mean"]}
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