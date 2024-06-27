import numpy as np
from sklearn.manifold import TSNE
from sklearn import decomposition
from sklearn.metrics import davies_bouldin_score, silhouette_score
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm, colors
import pandas as pd
from torchvision import datasets#, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
#from torch.utils.data import Subset

from brian2 import *
from umap import UMAP
from image_to_image import SAE
#from model.aux.functions import get_poisson_inputs, process_labels, mse_count_loss

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
from snntorch import utils
from random import random

__all__ = ["PoissonTransform",
           "build_datasets",
           "build_network",
           "to_np",
           "plot_input",
           "curr_to_pA",
           "transfer",
           "get_fr",
           "set_seed",
           "umap_plt",
           "get_poisson_inputs",
           "process_labels",
           "rmse_count_loss"]

class PoissonTransform(torch.nn.Module):
    def __init__(self, total_time, bin_size, rate_on, rate_off):
        super().__init__()
        self.total_time = total_time
        self.bin_size = bin_size
        self.rate_on = rate_on
        self.rate_off = rate_off
        
    def forward(self, img):
        new_image = get_poisson_inputs(img, self.total_time, self.bin_size, self.rate_on, self.rate_off)
        #print(f"Converted {img.shape} to {new_image.shape}")
        return new_image

def build_datasets(train_specs, input_specs = None):
    batch_size = train_specs["batch_size"]
    subset_size = train_specs["subset_size"]
    num_workers = train_specs["num_workers"]
    persist = True if num_workers > 0 else False
    
    if input_specs:
        print("Applying Poisson Transform")
        total_time = input_specs["total_time"]
        bin_size = input_specs["bin_size"]
        rate_on = input_specs["rate_on"]
        rate_off = input_specs["rate_off"]
        
        transform = v2.Compose([
                    v2.Grayscale(),
                    v2.ToTensor(),
                    v2.Normalize((0,), (1,)),
                    PoissonTransform(total_time, bin_size, rate_on, rate_off)
                    ])
    else:
        print("Not Applying Poisson Transform")
        transform = v2.Compose([
                    v2.Grayscale(),
                    v2.ToTensor(),
                    v2.Normalize((0,), (1,)),
                    ])
    
    # create dataset in /content
    print("\nMaking datasets and defining subsets")
    train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transform, download=True)
    
    trainlen1 = len(train_dataset)
    testlen1 = len(test_dataset)
    snn.utils.data_subset(train_dataset, subset_size)
    snn.utils.data_subset(test_dataset, subset_size)
    trainlen2 = len(train_dataset)
    testlen2 = len(test_dataset)
    
    print(f"Training: {trainlen1} -> {trainlen2}\nTesting: {testlen1} -> {testlen2}")
    print("\nMaking Dataloaders")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, persistent_workers=persist)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, persistent_workers=persist)

    return train_dataset, train_loader, test_dataset, test_loader
    
def build_network(device, noise = 0, recurrence = 1, num_rec = 100):
    print("Defining network")
    time_params, network_params, frame_params, convolution_params = {}, {}, {}, {}
    
    # Parameters for use in network definition
    time_params["dt"] = 1*ms
    time_params["total_time"] = 200*ms

    network_params["tau_m"] = 24*ms     # affects beta
    network_params["tau_syn"] = 10*ms   # not currently used
    network_params["R_m"] = 146*Mohm    # not currently used
    network_params["v_th"] = 1          # snn default = 1
    network_params["eta"] = noise        # controls noise amplitude - try adding noise in rec layer
    network_params["num_rec"] = num_rec
    network_params["num_latent"] = 8

    frame_params["depth"] = 1
    frame_params["size"] = 28

    convolution_params["channels_1"] = 12
    convolution_params["filter_1"] = 3
    convolution_params["channels_2"] = 64
    convolution_params["filter_2"] = 3

    network = SAE(time_params, network_params, frame_params, convolution_params, device, recurrence).to(device)
    
    return network, network_params
    
def to_np(tensor):
    return tensor.detach().cpu().numpy()

def plot_input(inputs, index):
    plt.imshow(to_np(torch.transpose(torch.sum(inputs, 0)[index], 0 ,2)))

def curr_to_pA(curr, network):
    factor = network.network_params["v_th"]/network.network_params["R_m"]/(1 - network.network_params["beta"])
    try:
      return to_np(curr)*factor
    except:
      return curr*factor

def transfer(curr, network):
    T = -network.network_params["tau_m"]*np.log(1 - network.network_params["v_th"]/(curr*network.network_params["R_m"]))
    return np.clip(1/T, 0*Hz, inf*Hz)

def get_fr(raster, network):
    return to_np(torch.sum(raster, 0))/network.time_params["total_time"]

def set_seed(value = 42):
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(value)
    print(f"\nSetting Seed to {value}")

def umap_plt(file, w=6, h=6):
    features = pd.read_csv(file)
    all_labs = features.iloc[:, 1]
    features = features.iloc[:, 2:-1]
    tail = os.path.split(file)
    f_name = f"UMAPS/umap_{tail[1]}.png"
    print("Applying UMAP")
    umap = UMAP(n_components=2).fit_transform(features)
    cmap = mpl.colormaps['viridis']
    plt.figure(figsize=(w, h))
    c_range = np.arange(0.5, 10, 1)
    norm = colors.BoundaryNorm(c_range, cmap.N)
    plt.scatter(umap[:, 0], umap[:, 1], c=all_labs, cmap=cmap, norm=norm)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.colorbar(label='Digit Class', ticks=c_range-0.5)
    plt.savefig(f_name)
    plt.title(tail[1])
    plt.show()
    
    print("Calculating Cluster Scores - S/D-B")
    sil_score = silhouette_score(umap, all_labs)
    db_score = davies_bouldin_score(umap, all_labs)
    
    return f_name, sil_score, db_score
    
def get_poisson_inputs(inputs, total_time, bin_size, rate_on, rate_off):
    num_steps = int(total_time/bin_size)
    bin_prob_on = rate_on*bin_size # 75 Hz * 1ms = 0.075
    bin_prob_off = rate_off*bin_size # 10 Hz * 1ms - 0.010
    poisson_input = snn.spikegen.rate((bin_prob_on - bin_prob_off)*inputs + bin_prob_off*torch.ones(inputs.shape) , num_steps=num_steps) # default: inputs = data
    return poisson_input

class rmse_count_loss():
    def __init__(self, lambda_rate, lambda_weights):  
        self.lambda_r = lambda_rate
        self.lambda_w = lambda_weights
        self.__name__ = "rmse_count_loss"
        
    def __call__(self, spk_recs, outputs, inputs):
        spike_count = torch.sum(outputs, 0)
        target_spike_count = torch.sum(inputs, 0)
        loss_fn = nn.MSELoss()
        loss = torch.sqrt(loss_fn(spike_count, target_spike_count)) + self.lambda_r*torch.sum(spk_recs)
        return loss

'''
Implements the normalised MSE by dividing the MSE by the sum  of squares of the input.
This is equivalent to normalising the RMSE by the L2 Norm of the input.
'''    
class nmse_count_loss():
    def __init__(self, lambda_rate=0, ntype = None):  
        self.lambda_r = lambda_rate
        self.__name__ = "nrmse_count_loss"
        self.ntype = ntype
    def __call__(self, outputs, inputs, spk_recs=torch.tensor(0)):
        
        # make it agnostic to spiking or non-spiking tensors
        spike_count = torch.sum(outputs, 0) if outputs.dim() > 4 else outputs
        target_spike_count = torch.sum(inputs, 0) if inputs.dim() > 4 else inputs
        loss_fn = nn.MSELoss()
        
        if self.ntype == None:
            loss = torch.sqrt(loss_fn(spike_count, target_spike_count)) + self.lambda_r*torch.sum(spk_recs)
        
        if self.ntype == "norm":
            loss_fn = nn.MSELoss(reduction="sum")
            loss = loss_fn(spike_count, target_spike_count)/loss_fn(torch.zeros_like(spike_count), target_spike_count) + self.lambda_r*torch.sum(spk_recs)
        
        elif self.ntype == "range":
            loss = torch.sqrt(loss_fn(spike_count, target_spike_count))/(torch.max(target_spike_count) - torch.min(target_spike_count)) + self.lambda_r*torch.sum(spk_recs)
        
        elif self.ntype == "mean":
            loss = torch.sqrt(loss_fn(spike_count, target_spike_count))/torch.mean(target_spike_count) + self.lambda_r*torch.sum(spk_recs)      
        
        else:
            Exception("Enter valid string: norm, range, or mean.")
        
        return loss
    
class mae_count_loss():
    def __init__(self, lambda_rate=0):  
        self.lambda_r = lambda_rate
        self.__name__ = "rmse_count_loss"
        
    def __call__(self, spk_recs, outputs, inputs):
        spike_count = torch.sum(outputs, 0)
        target_spike_count = torch.sum(inputs, 0)
        loss_fn = nn.L1Loss()
        loss = torch.sqrt(loss_fn(spike_count, target_spike_count)) + self.lambda_r*torch.sum(spk_recs)
        return loss