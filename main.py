"""## Imports"""
#print("\nImporting modules and defining functions")

#importing module
from datetime import datetime
date = datetime.now().strftime("%d/%m - %H:%M")

import wandb
from helpers import build_datasets, build_network, set_seed, build_nmnist_dataset, plotting_data

import torch
from brian2 import *
from train_network import train_network, train_network_bptt

import gc
    
def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using CUDA") if torch.cuda.is_available() else print("Using CPU")
    set_seed()
    torch.backends.cudnn.benchmark = True #TURN OFF WHEN CHANGING ARCHITECTURE    
    run = wandb.init()
    
    """## Define network architecutre and parameters"""
    network_params, input_specs, train_specs = {}, {}, {}
    
    # Parameters for use in training
    input_specs["rate_on"] = wandb.config.rate_on*Hz
    input_specs["rate_off"] = wandb.config.rate_off*Hz    
    input_specs["total_time"] = 100*ms
    input_specs["bin_size"] = 1*ms
    
    train_specs["code"] = 'rate'
    
    train_specs["early_stop"] = -1
    train_specs["loss_fn"] = "spike_count"
    train_specs["lambda_rate"] = 0.0
    train_specs["lambda_weights"] = None
    train_specs["num_epochs"] = wandb.config.epochs    
    train_specs["device"] = device
    train_specs["lr"] = wandb.config.lr    
    train_specs["batch_size"] = wandb.config.bs# if not wandb.config.first_saccade and not wandb.config.MNIST else 16
    train_specs["subset_size"] = wandb.config.subset_size
    train_specs["num_workers"] = wandb.config.num_workers
    train_specs["norm_type"] = wandb.config.norm_type
    num_rec = wandb.config.num_rec
    learnable = wandb.config.learnable
    noise = wandb.config.noise
    recurrence = wandb.config.recurrence
    MNIST = wandb.config.MNIST
    first_saccade_only = wandb.config.first_saccade
    
    print(f'Starting Sweep: Batch Size: {train_specs["batch_size"]}, Learning Rate: {train_specs["lr"]}')
    
    # Build dataset and loaders
    
    if MNIST:
        dataset = "MNIST"
        train_dataset, train_loader, test_dataset, test_loader = build_datasets(train_specs, input_specs)
    else:
        dataset = "NMNIST"
        train_dataset, train_loader, test_dataset, test_loader = build_nmnist_dataset(train_specs, first_saccade=first_saccade_only)
    
    run.name = f"{dataset}_{wandb.config.first_saccade}fs_{wandb.config.bs}_{wandb.config.lr}"
    
    in_size = train_dataset[0][0].shape[-1]
    
    # Build network
    network, network_params = build_network(device, noise=noise, recurrence=recurrence, num_rec=num_rec, learnable=learnable, size=in_size)
    
    # Train network
    network, train_loss, test_loss, final_train_loss, final_test_loss = train_network(network, train_loader, test_loader, train_specs)
    
    labels, umap_file, sil_score, db_score = plotting_data(network, train_dataset, test_dataset, train_loader, test_loader, recurrence, device, run)
      
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
    
    del network, train_loss, test_loss, final_train_loss, final_test_loss
        
    gc.collect()
    torch.cuda.empty_cache()
        
        
if __name__ == '__main__':  
  
  test = 1
  
  if test == 1:
      sweep_config = {
          'name': f'Test Sweep (Mem Leaks) {date}',
          'method': 'grid',
          'metric': {'name': 'Test Loss',
                      'goal': 'minimize'   
                      },
          'parameters': {'bs': {'values': [64]},
                          'lr': {'values': [1e-4]},
                          'epochs': {'values': [9]},
                          "subset_size": {'values': [10]},
                          "recurrence": {'values': [True]},
                          "noise": {'values': [0]},
                          "rate_on": {'values': [75]},
                          "rate_off": {'values': [1]},
                          "num_workers": {'values': [0]},
                          "num_rec": {'values': [100]},
                          "norm_type": {'values': ["norm"]},
                          "learnable": {'values': [True]},
                          "MNIST": {'values': [True]},
                          "first_saccade": {'values': [True]}
                          }
          }
  else:
      sweep_config = { #REMEMBER TO CHANGE RUN NAME
          'name': f'NMNIST recurrence and learnable {date}',
          'method': 'grid',
          'metric': {'name': 'Test Loss',
                      'goal': 'minimize'   
                      },
          'parameters': {'bs': {'values': [16]},
                          'lr': {'values': [1e-4]},
                          'epochs': {'values': [9]},
                          "subset_size": {'values': [10]},
                          "recurrence": {'values': [1, 0]},
                          "noise": {'values': [0]},
                          "rate_on": {'values': [75]},
                          "rate_off": {'values': [1]},
                          "num_workers": {'values': [0]},
                          "num_rec": {'values': [100]},
                          "norm_type": {'values': ["norm"]},
                          "learnable": {'values': [True, False]},
                          "MNIST": {'values': [False]},
                          "first_saccade": {'values': [False]}
                          }
          }
  
  
  sweep_id = wandb.sweep(sweep = sweep_config, project = "MSc Project", entity="lukelalderson")
      
  wandb.agent(sweep_id, function=main)
  
  torch.cuda.empty_cache()