import sys
sys.path.append('snn-project')
import random
import numpy as np
import wandb
from sklearn.manifold import TSNE
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import utils as utls
from torch.utils.data import Subset

import snntorch as snn
import snntorch.spikeplot as splt
from snntorch import utils
from snntorch import surrogate

from IPython.display import HTML
from brian2 import *
import seaborn as sns

from model.train.train_network import train_network
from model.image_to_image import SAE
from model.aux.functions import get_poisson_inputs, process_labels, mse_count_loss

from math import floor, ceil
from datetime import datetime

__all__ = ["build_datasets", "build_network", "to_np", "plot_input", \
           "curr_to_pA", "transfer", "get_fr", "print_network_architecure", \
           "set_seed", "tsne_plt"]

def build_datasets(train_specs): # add subsampling parameeter
        """## Make or access existing datasets"""
        
        batch_size = train_specs["batch_size"]
        subset_size = train_specs["subset_size"]
        num_workers = train_specs["num_workers"]
        
        persist = True if num_workers > 0 else False
        
        if subset_size > 1 or subset_size < 0:
            raise Exception("Subset must be a floating number between 0 and 1.")
               
        transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0,), (1,))])
        
        # create dataset in /content
        print("\nMaking datasets and defining subsets")
        train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transform, download=True)
        

        train_subset_size = round(subset_size*len(train_dataset))
        test_subset_size = round(subset_size*len(test_dataset))
        print(f"Training: {len(train_dataset)} -> {train_subset_size}\nTesting: {len(test_dataset)} -> {test_subset_size}")
       
        if subset_size != 1:     
            print("\nMaking Subsets")
            subset_train_indices = torch.randperm(len(train_dataset))[:train_subset_size]
            subset_test_indices = torch.randperm(len(test_dataset))[:test_subset_size]
            train_dataset = Subset(train_dataset, subset_train_indices)
            test_dataset = Subset(test_dataset, subset_test_indices)
            print(f"Training: {len(train_dataset)}\nTesting: {len(test_dataset)}")
        print("\nMaking Dataloaders")
        # snn.utils.data_subset(train_dataset, subset_size)
        # snn.utils.data_subset(test_dataset, subset_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, persistent_workers=persist)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, persistent_workers=persist)
    
        return train_dataset, train_loader, test_dataset, test_loader
    
def build_network(device, noise = 0, recurrence = True):
    print("Defining network")
    time_params, network_params, oscillation_params, frame_params, \
    convolution_params, input_specs, label_specs, train_specs = {}, {}, {}, {}, {}, {}, {}, {}
    # Parameters for use in network definition
    time_params["dt"] = 1*ms
    time_params["total_time"] = 200*ms

    network_params["tau_m"] = 24*ms     # affects beta
    network_params["tau_syn"] = 10*ms   # not currently used
    network_params["R_m"] = 146*Mohm    # not currently used
    network_params["v_th"] = 1          # snn default = 1
    network_params["eta"] = noise        # controls noise amplitude - try adding noise in rec layer
    network_params["num_rec"] = 100
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

def print_network_architecure(network):
    netp, op, fp, cp = network.network_params, network.oscillation_params, network.frame_params, network.convolution_params
    input_layer_text = """
    Input layer: {} channels
                {}x{} neurons/channel
                {} total neurons
    """.format(fp["depth"], fp["size"], fp["size"],fp["depth"]*fp["size"]*fp["size"] )

    conv1_text = """
    Conv1 layer: {} channels
                {}x{} neurons/channel
                {} total neurons
                {} synapses/neuron (ff)
                {} total_synapses
    """.format(cp["channels_1"], cp["conv1_size"], cp["conv1_size"], netp["num_conv1"], cp["filter_1"]*cp["filter_1"], netp["num_conv1"]*cp["filter_1"]*cp["filter_1"])

    conv2_text = """
    Conv2 layer: {} channels
                {}x{} neurons/channel
                {} total neurons
                {} synapses/neuron (ff)
                {} total_synapses
    """.format(cp["channels_2"], cp["conv2_size"], cp["conv2_size"], netp["num_conv2"], cp["filter_2"]*cp["filter_2"], netp["num_conv2"]*cp["filter_2"]*cp["filter_2"])

    rec_text = """
    Rec layer:   {} total neurons
                {} synapses/neuron (ff) and {} synapses/neuron (rec)
                {} total_synapses
    """.format(netp["num_rec"], netp["num_conv2"], netp["num_rec"], netp["num_conv2"]*netp["num_rec"] + netp["num_rec"]**2)

    latent_text = """
    Lat layer:   {} total neurons
                {} synapses/neuron (ff)
                {} total_synapses
    """.format(netp["num_latent"], netp["num_rec"], netp["num_rec"], netp["num_rec"]*netp["num_latent"])

    Trec_text = ""
    Tconv2_text = ""
    Tconv1_text = ""
    output_layer_text = ""

    print(input_layer_text)
    print(conv1_text)
    print(conv2_text)
    print(rec_text)
    print(latent_text)
    print(Trec_text)
    print(Tconv2_text)
    print(Tconv1_text)
    print(output_layer_text)
    

def set_seed(value = 42):
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    #random.seed(value)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(value)
    print(f"\nSetting Seed to {value}")
    
def tsne_plt(file):
    features = pd.read_csv(file)
    features = features.iloc[:, 1:-1]
    print("Applying t-SNE")
    tsne = TSNE().fit_transform(features)
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=all_labs, cmap='viridis')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(label='Digit Class')
    plt.savefig("tsne.png")
    plt.show()