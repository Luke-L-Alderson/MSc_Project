import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score
import os
import matplotlib as mpl
from matplotlib import cm, colors
from matplotlib import pyplot as plt
import pandas as pd
from torchvision import datasets#, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset, Subset
#from torch.utils.data import Subset
import seaborn as sns
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

import tonic.transforms as transforms
import tonic

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
    
class dtype_transform(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        
    def forward(self, img):
        #print(img[:, 1].unsqueeze(1).shape)
        return img[:, 0].unsqueeze(1).type(torch.FloatTensor)

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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, persistent_workers=persist, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, persistent_workers=persist, collate_fn=custom_collate_fn)
    
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
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, persistent_workers=persist)
    

    return train_dataset, train_loader, test_dataset, test_loader
    
def build_network(device, noise = 0, recurrence = 1, num_rec = 100, learnable=True, depth=1, size=28, time=200):
    print("Defining network")
    time_params, network_params, frame_params, convolution_params = {}, {}, {}, {}
    
    # Parameters for use in network definition
    time_params["dt"] = 1*ms
    time_params["total_time"] = time*ms

    network_params["tau_m"] = 24*ms     # affects beta
    network_params["tau_syn"] = 10*ms   # not currently used
    network_params["R_m"] = 146*Mohm    # not currently used
    network_params["v_th"] = 1          # snn default = 1
    network_params["eta"] = noise        # controls noise amplitude - try adding noise in rec layer
    network_params["num_rec"] = num_rec
    network_params["learnable"] = learnable
    frame_params["depth"] = depth
    frame_params["size"] = size

    convolution_params["channels_1"] = 12
    convolution_params["filter_1"] = 3
    convolution_params["channels_2"] = 64
    convolution_params["filter_2"] = 3

    network = SAE(time_params, network_params, frame_params, convolution_params, device, recurrence).to(device)
    
    for name, param in network.named_parameters():
        print(f"{name} --> {param.shape}")
    
    try:
        fig = plt.figure(facecolor="w", figsize=(10, 10))
        
        ax1 = plt.subplot(2, 2, 1)
        weight_map(network.rlif_rec.recurrent.weight)
        plt.title("Initial Weights")
        
        ax2 = plt.subplot(2, 2, 3)
        sns.histplot(to_np(torch.flatten(network.rlif_rec.recurrent.weight)))
        plt.title("Initial Weight Distribution")
        
        #network.rlif_rec.recurrent.weight = nn.Parameter(1*torch.ones_like(network.rlif_rec.recurrent.weight))
        
        ax3 = plt.subplot(2, 2, 2)
        weight_map(network.rlif_rec.recurrent.weight)
        plt.title("Adjusted Weights")
        
        ax4 = plt.subplot(2, 2, 4)
        sns.histplot(to_np(torch.flatten(network.rlif_rec.recurrent.weight)))
        plt.title("Adjusted Weight Distribution")
        
        plt.show() 
    
    except:
        print("Not recurrent")

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
    try:
        random.seed()
    except:
        print("Couldn't set random.seed()")
    print(f"\nSetting Seed to {value}")

def umap_plt(file, w=10, h=5):
    features = pd.read_csv(file)
    all_labs = features["Labels"]#.to_numpy()
    #print(all_labs)
    features = features.loc[:, features.columns != 'Labels']#.to_numpy()
    #print(f"Printing Features: \n{features.iloc[0, :]}")
    tail = os.path.split(file)
    f_name = f"UMAPS/umap_{tail[1]}.png"
    print("Applying UMAP")
    umap = UMAP(n_components=2, random_state=42, n_neighbors=15).fit_transform(features)
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
    sil_score = silhouette_score(umap[:, 0:1], all_labs)
    db_score = davies_bouldin_score(umap[:, 0:1], all_labs)
    
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
        loss_fn = nn.MSELoss() # include max and min rates
        
        if self.ntype == None:
            loss = torch.sqrt(loss_fn(spike_count, target_spike_count)) + self.lambda_r*torch.sum(spk_recs)
        
        if self.ntype == "norm":
            loss_fn = nn.MSELoss(reduction="sum")
            loss = torch.sqrt(loss_fn(spike_count, target_spike_count)/loss_fn(torch.zeros_like(spike_count), target_spike_count)) + self.lambda_r*torch.sum(spk_recs)
        
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

def weight_map(wm, w=10, h=10, sign=False): # wm should be a tensor of weights
    #fig = plt.figure(facecolor="w", figsize=(w, h))
    weight_log = np.sign(to_np(wm)) if sign else to_np(wm)
    num_rec = wm.shape[0]
    ax = sns.heatmap(weight_log)
    plt.xlabel('# Neuron (Layer Output)')
    plt.ylabel('# Neuron (Layer Input)')
    ax.invert_yaxis()
    plt.xlim([0, num_rec])
    plt.ylim([0, num_rec])
    ax.set_xticks(np.arange(0, num_rec+1, num_rec/10), labels=np.arange(0, num_rec+1, num_rec/10))
    ax.set_yticks(np.arange(0, num_rec+1, num_rec/10), labels=np.arange(0, num_rec+1, num_rec/10))
    return ax

def create_subset(dataset, subset_size):
    indices = np.random.choice(len(dataset), subset_size, replace=False)
    subset=Subset(dataset, indices)
    return subset

'''
Inputs: train_specs, input_specs [Optional]

Outputs: Dataloaders
         Datasets, where each element is a tuple (data, label), and data is a tensor.
'''
def build_nmnist_dataset(train_specs, input_specs = None):
    num_workers = train_specs["num_workers"]
    batch_size = train_specs["batch_size"]
    sensor_size = tonic.datasets.NMNIST.sensor_size
    subset_size = train_specs["subset_size"]
    
    persist = True if num_workers > 0 else False
    
    raw_transform = tonic.transforms.Compose([
                transforms.Denoise(filter_time=10000),
                transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
                torch.from_numpy,
                dtype_transform()
                ])
    
    print("\nMaking datasets and defining subsets")
    train_dataset = tonic.datasets.NMNIST(save_to='./dataset',
                                          transform=raw_transform,
                                          train=True,
                                          first_saccade_only=True
                                          )
    
    test_dataset = tonic.datasets.NMNIST(save_to='./dataset',
                                          transform=raw_transform,
                                          train=False,
                                          first_saccade_only=True)
    
    trainlen1 = len(train_dataset)
    testlen1 = len(test_dataset)
     
    train_dataset = create_subset(train_dataset, int(len(train_dataset)/subset_size))
    test_dataset = create_subset(test_dataset, int(len(test_dataset)/subset_size))
    
    trainlen2 = len(train_dataset)
    testlen2 = len(test_dataset)
    
    print(f"Training: {trainlen1} -> {trainlen2}\nTesting: {testlen1} -> {testlen2}")
    print("\nMaking Dataloaders")
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              collate_fn=tonic.collation.PadTensors(batch_first=False), 
                              shuffle=True, 
                              pin_memory=True, 
                              num_workers=num_workers, 
                              persistent_workers=persist)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size, 
                             collate_fn=tonic.collation.PadTensors(batch_first=False), 
                             pin_memory=True, 
                             num_workers=num_workers, 
                             persistent_workers=persist)
    
    
    return train_dataset, train_loader, test_dataset, test_loader

def custom_collate_fn(batch):
    # Unpack the batch
    images, labels = zip(*batch)
    
    # Stack the images and labels
    # images is a list of tensors of shape [t, 1, 28, 28]
    images = torch.stack(images)  # Shape: [bs, t, 1, 28, 28]
    labels = torch.tensor(labels) # Shape: [bs]
    # Permute the images to the desired shape [t, bs, 1, 28, 28]
    images = images.transpose(0, 1)  # New shape: [t, bs, 1, 28, 28]
    
    return images, labels