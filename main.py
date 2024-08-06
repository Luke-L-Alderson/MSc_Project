"""## Imports"""
#print("\nImporting modules and defining functions")

#importing module
from datetime import datetime
date = datetime.now().strftime("%d/%m - %H:%M")

import wandb
from helpers import build_datasets, build_network, set_seed, \
build_nmnist_dataset, plotting_data, visTensor, weight_map, to_np,\
build_dvs_dataset
from matplotlib import pyplot as plt
import torch
import torchvision.models as models
from brian2 import *
from train_network import train_network, train_network_bptt
import seaborn as sns
import gc
    
def main():
    
    # for obj in gc.get_objects():
    #     try:
    #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #             print(type(obj), obj.size())
    #     except:
    #         pass
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using CUDA") if torch.cuda.is_available() else print("Using CPU")
    
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
    
    run.name = f"{wandb.config.recurrence}_{wandb.config.kernel_size}"
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
    
    #visTensor(network)
    
    # Train network
    network, train_loss, test_loss, final_train_loss, final_test_loss = train_network(network, train_loader, test_loader, train_specs, k1=k1)
    
    #visTensor(network)
    
    torch.save(network.state_dict(), f'SAE_{run.name}.pth')
    
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
  
  test = 0
  
  numrecs = list(range(490, 511, 2)) + list(range(590, 611, 2))
  
  if test == 1:
      sweep_config = {
          'name': f'Test Sweep (DVS) {date}',
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
                          "rate_on": {'values': [75]},
                          "rate_off": {'values': [1]},
                          "num_workers": {'values': [0]},
                          "num_rec": {'values': [100]},
                          "norm_type": {'values': ["mean"]},
                          "MNIST": {'values': [True]},
                          "first_saccade": {'values': [True]},
                          "kernel_size": {'values': [7]},
                          "seed": {'values': [42]},
                          "k1": {'values': [20]}
                          }
          }
  else:
      sweep_config = { #REMEMBER TO CHANGE RUN NAME
          'name': f'Recurrence Assessment (MNIST) {date}',
          'method': 'grid',
          'metric': {'name': 'Test Loss',
                      'goal': 'minimize'   
                      },
          'parameters': {'bs': {'values': [64]},
                          'lr': {'values': [1e-4]},
                          'epochs': {'values': [10]},
                          "subset_size": {'values': [10]},
                          "recurrence": {'values': [True, False]},
                          "noise": {'values': [0]},
                          "rate_on": {'values': [75]},
                          "rate_off": {'values': [1]},
                          "num_workers": {'values': [0]},
                          "num_rec": {'values': [100]}, # 100, 2, 50, 200, 500, 1000
                          "norm_type": {'values': ["mean"]},
                          "MNIST": {'values': [True]},
                          "first_saccade": {'values': [True]},
                          "kernel_size": {'values': [3, 5, 7]},
                          "seed": {'values': [42, 28, 1997, 1984, 1]}, # 42, 28, 1997, 1984, 1
                          "k1": {'values': [20]} #k1 = 25, bs = 64, morm=None worked
                          }
          }
  
  
  '''
    {'bs': {'values': [64]},
                    'lr': {'values': [1e-4]},
                    'epochs': {'values': [10]},
                    "subset_size": {'values': [1]},
                    "recurrence": {'values': [True, False]},
                    "noise": {'values': [0]},
                    "rate_on": {'values': [75]},
                    "rate_off": {'values': [1]},
                    "num_workers": {'values': [0]},
                    "num_rec": {'values': [100]}, # 100, 2, 50, 200, 500, 1000
                    "norm_type": {'values': ["mean"]},
                    "MNIST": {'values': [True]},
                    "first_saccade": {'values': [True]},
                    "kernel_size": {'values': [3, 5, 7]},
                    "seed": {'values': [42, 28, 1997, 1984, 1]}, # 42, 28, 1997, 1984, 1
                    "k1": {'values': [20]} #k1 = 25, bs = 64, morm=None worked
                    }
    '''
    
  sweep_id = wandb.sweep(sweep = sweep_config, project = "MSc Project", entity="lukelalderson")
      
  wandb.agent(sweep_id, function=main)
  
  torch.cuda.empty_cache()