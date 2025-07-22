"""## Imports"""
#print("\nImporting modules and defining functions")

#importing module
from datetime import datetime
date = datetime.now().strftime("%d/%m - %H:%M")

import wandb
from helpers import build_datasets, build_network, set_seed, plotting_data, weight_map, to_np, build_dvs_dataset
from matplotlib import pyplot as plt
import torch
import torchvision.models as models
from brian2 import *
from train_network import train_network, train_network_bptt, train_network_c
import seaborn as sns
import gc
    
def main():
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    torch.backends.cudnn.benchmark = True #TURN OFF WHEN CHANGING ARCHITECTURE    
    run = wandb.init()
    set_seed(wandb.config.seed)
    
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
    train_specs["lambda_rate"] = wandb.config.lmbda
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
    MNIST = wandb.config.MNIST
    first_saccade_only = wandb.config.first_saccade
    kernel_size = wandb.config.kernel_size
    k1 = wandb.config.k1
    print(f'Starting Sweep: Batch Size: {train_specs["batch_size"]}, Learning Rate: {train_specs["lr"]}')
    
    # Build dataset and loaders
    
    if MNIST:
        dataset = "MNIST"
        train_dataset, train_loader, test_dataset, test_loader = build_datasets(train_specs, input_specs)
    else:
        dataset = "DVS"
        train_dataset, train_loader, test_dataset, test_loader = build_dvs_dataset(train_specs, first_saccade=first_saccade_only)
    
    run.name = f"{wandb.config.rate_on}_{wandb.config.rate_off}"
    #run.name = "UMAP"
    
    in_size = train_dataset[0][0].shape[-1]
    
    # Build network
    network, network_params = build_network(device, noise=noise, recurrence=recurrence, num_rec=num_rec, size=in_size, kernel_size=kernel_size)
    
    if recurrence == True:
        fig = plt.figure(facecolor="w", figsize=(10, 5))
        
        ax1 = plt.subplot(1, 2, 1)
        weight_map(network.rlif_rec.recurrent.weight)
        plt.title("Initial Weight Heatmap")
        
        ax2 = plt.subplot(1, 2, 2)
        sns.histplot(to_np(torch.flatten(network.rlif_rec.recurrent.weight)))
        plt.title("Initial Weight Distribution")
        
        plt.show()
       
    # Train network
    network, train_loss, test_loss, final_train_loss, final_test_loss = train_network(network, train_loader, test_loader, train_specs, k1=k1, state=wandb.config.state)
    
    #torch.save(network.state_dict(), f'SAE_{run.name}.pth')
    
    # Plotting
    labels, umap_file, sil_score, db_score = plotting_data(network, train_dataset, test_dataset, train_loader, test_loader, recurrence, device, run, k1=k1)
    
    # Logging
    wandb.log({"Test Loss": final_test_loss,
                "Results Grid": wandb.Image(f"figures/result_summary_{run.name}.png"),
                "UMAP": wandb.Image(umap_file),
                "Spike Animation": wandb.Video(f"figures/spike_mnistrec_{labels}.gif", fps=4, format="gif"),
                "Raster": wandb.Image("figures/rasters.png"),
                "Time Slices": wandb.Image(f"figures/time_slices_{run.name}.png"),
                "Sil Score": sil_score,
                "DB Score": db_score,
                })
    
    try:
        wandb.log({"Weights": wandb.Image("figures/weightmap.png")})
    except:
        pass
    
    del network, train_loss, test_loss, final_train_loss, final_test_loss, \
        train_dataset, train_loader, test_dataset, test_loader
        
    gc.collect()
    torch.cuda.empty_cache()
        
        
if __name__ == '__main__':  
  
  dvs = False
  
  numrecs = list(range(490, 511, 2)) + list(range(590, 611, 2))
  seeds = [42, 28, 1997, 1984, 1, 2, 100, 200, 800, 500]
  
  if dvs:
      sweep_config = {
          'name': f'Test Sweep (DVS classifier) {date}',
          'method': 'grid',
          'metric': {'name': 'Test Loss',
                      'goal': 'minimize'   
                      },
          'parameters': {'bs': {'values': [64]},
                          'lr': {'values': [1e-4]},
                          'epochs': {'values': [30]},
                          "subset_size": {'values': [1]},
                          "recurrence": {'values': [False]},
                          "noise": {'values': [0]},
                          "rate_on": {'values': [75]},
                          "rate_off": {'values': [1]},
                          "num_workers": {'values': [0]},
                          "num_rec": {'values': [1000]},
                          "norm_type": {'values': ["mean"]},
                          "MNIST": {'values': [False]},
                          "first_saccade": {'values': [True]},
                          "kernel_size": {'values': [7]},
                          "seed": {'values': [42]},
                          "k1": {'values': [20]}, # 42, 28, 1997, 1984, 1
                          "lmbda": {'values': [0]}, # 42, 28, 1997, 1984, 1
                          "state": {'values': ["present"]}
                          }
          }
  else:
      sweep_config = { 
          'name': f'MNIST freq scan 3 {date}',
          'method': 'grid',
          'metric': {'name': 'Test Loss',
                      'goal': 'minimize'   
                      },
          'parameters': {'bs': {'values': [64]},
                          'lr': {'values': [1e-4]},
                          'epochs': {'values': [10]},
                          "subset_size": {'values': [10]},
                          "recurrence": {'values': [False]},
                          "noise": {'values': [0]},
                          "rate_on": {'values': [1, 50, 75, 100, 150, 200]},
                          "rate_off": {'values': [1]},
                          "num_workers": {'values': [0]},
                          "num_rec": {'values': [100, 600]},# 550, 500, 600, 100
                          "norm_type": {'values': ["mean"]},
                          "MNIST": {'values': [True]},
                          "first_saccade": {'values': [True]},
                          "kernel_size": {'values': [7]},
                          "seed": {'values': [42]},        # 42, 28, 1997, 1984, 1
                          "k1": {'values': [20]},          # 42, 28, 1997, 1984, 1
                          "lmbda": {'values': [0]},        # 42, 28, 1997, 1984, 1
                          "state": {'values': ["present"]} # k1 = 25, bs = 64, morm=None worked
                          }
          }
    
  sweep_id = wandb.sweep(sweep = sweep_config, project = "MSc Project", entity="lukelalderson")
      
  wandb.agent(sweep_id, function=main)
  
  torch.cuda.empty_cache()