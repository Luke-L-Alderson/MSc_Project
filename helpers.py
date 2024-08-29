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
import wandb
from math import ceil
import seaborn as sns
from brian2 import *
from umap import UMAP
from image_to_image import SAE, SAE_ni, SC
#from model.aux.functions import get_poisson_inputs, process_labels, mse_count_loss
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
from snntorch import utils
from snntorch import functional as F
import torchvision.utils as utls
import snntorch.spikeplot as splt
from IPython.display import HTML
import tonic.transforms as transforms
import tonic
from torchsummary import summary
import random as rand
from tonic import DiskCachedDataset
import warnings
import gc
from snntorch import utils
from math import ceil

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
        return new_image
    
class dtype_transform(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        
    def forward(self, img):
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
    
def build_network(device, noise = 0, recurrence = True, num_rec = 100, depth=1, size=28, time=200, kernel_size = 3):
    print("Defining network")
    time_params, network_params, frame_params, convolution_params = {}, {}, {}, {}
    print(f"Input Size is {size}")
    # Parameters for use in network definition
    time_params["dt"] = 1*ms
    time_params["total_time"] = time*ms

    network_params["tau_m"] = 24*ms     # affects beta
    network_params["tau_syn"] = 10*ms   # not currently used
    network_params["R_m"] = 146*Mohm    # not currently used
    network_params["v_th"] = 1          # snn default = 1
    network_params["eta"] = noise        # controls noise amplitude - try adding noise in rec layer
    network_params["num_rec"] = num_rec
    frame_params["depth"] = depth
    frame_params["size"] = size

    convolution_params["channels_1"] = 12
    convolution_params["filter_1"] = kernel_size
    convolution_params["channels_2"] = 64
    convolution_params["filter_2"] = kernel_size

    network = SAE(time_params, network_params, frame_params, convolution_params, device, recurrence).to(device)
    
    print_params(network)
    print(f"Total number of parameters is {count_params(network)}")

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
        rand.seed(value)
        print("Successfully set random.seed()")
    except:
        print("Couldn't set random.seed()")
    print(f"\nSetting Seed to {value}")

def umap_plt(file, w=10, h=5, s=1):
    features = pd.read_csv(file)
    all_labs = features["Labels"]#.to_numpy()
    num_labs = len(set(all_labs.to_numpy()))
    print(num_labs)
    #print(all_labs)
    features = features.loc[:, features.columns != 'Labels']#.to_numpy()
    #print(f"Printing Features: \n{features.iloc[0, :]}")
    tail = os.path.split(file)
    f_name = f"UMAPS/umap_{tail[1]}.png"
    print("Applying UMAP")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        umap = UMAP(n_components=2, random_state=42, n_neighbors=15).fit_transform(features)
    cmap = mpl.colormaps['viridis']
    fig = plt.figure(figsize=(w, h))
    #c_range = np.arange(0.5, 10, 1)
    c_range = np.arange(0.5, num_labs, 1)
    norm = colors.BoundaryNorm(c_range, cmap.N)
    ax = plt.scatter(umap[:, 0], umap[:, 1], c=all_labs, cmap=cmap, norm=norm, s=s)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.colorbar(label='Digit Class', ticks=c_range-0.5)
    plt.savefig(f_name)
    plt.title(tail[1])
    plt.show()
    
    print("Calculating Cluster Scores - S/D-B")
    sil_score = silhouette_score(umap[:, 0:1], all_labs)
    db_score = davies_bouldin_score(umap[:, 0:1], all_labs)
    
    return f_name, sil_score, db_score, ax
    
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
        # inputs = [t, bs, ch, h, w]
        
        # make it agnostic to spiking or non-spiking tensors - sums over time if available
        spike_count = torch.sum(outputs, 0) if outputs.dim() > 4 else outputs
        target_spike_count = torch.sum(inputs, 0) if inputs.dim() > 4 else inputs
        
        loss_fn = nn.MSELoss()
        
        if self.ntype == None: # RMSE Spike Count
            loss = torch.sqrt(loss_fn(spike_count, target_spike_count)) + self.lambda_r*torch.sum(spk_recs)  
        
        if self.ntype == "norm": # RMSE/"norm" Spike Count
            loss_fn = nn.MSELoss(reduction="sum")
            loss = torch.sqrt(loss_fn(spike_count, target_spike_count)/loss_fn(torch.zeros_like(spike_count), target_spike_count)) + self.lambda_r*torch.sum(spk_recs)
        
        elif self.ntype == "range": # RMSE/range Spike Count
            loss = torch.sqrt(loss_fn(spike_count, target_spike_count))/(torch.max(target_spike_count) - torch.min(target_spike_count)) + self.lambda_r*torch.sum(spk_recs)
        
        elif self.ntype == "mean": # RMSE/mean Spike Count
            loss = torch.sqrt(loss_fn(spike_count, target_spike_count))/torch.mean(target_spike_count) + self.lambda_r*torch.sum(spk_recs)      
        
        elif self.ntype == "spike_train": # RMSE/"norm" All Spikes
            loss = loss_fn(outputs, inputs) + self.lambda_r*torch.sum(spk_recs)
            
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

def weight_map(wm, w=10, h=10, sign=True): # wm should be a tensor of weights
    weight_log = np.sign(to_np(wm)) if sign else to_np(wm)
    num_rec = wm.shape[0]
    ax = sns.heatmap(weight_log)
    plt.xlabel('# Neuron (Layer Output)')
    plt.ylabel('# Neuron (Layer Input)')
    ax.invert_yaxis()
    plt.xlim([0, num_rec])
    plt.ylim([0, num_rec])
    labels = np.arange(0, num_rec+1, num_rec/10, dtype="int16")
    ax.set_xticks(np.arange(0, num_rec+1, num_rec/10), labels=labels)
    ax.set_yticks(np.arange(0, num_rec+1, num_rec/10), labels=labels)
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
def build_nmnist_dataset(train_specs, input_specs = None, first_saccade = False):
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
    
    first_saccade_only = first_saccade
    print("\nMaking datasets and defining subsets")
    train_dataset = tonic.datasets.NMNIST(save_to='./dataset',
                                          transform=raw_transform,
                                          train=True,
                                          first_saccade_only=first_saccade_only
                                          )
    
    test_dataset = tonic.datasets.NMNIST(save_to='./dataset',
                                          transform=raw_transform,
                                          train=False,
                                          first_saccade_only=first_saccade_only)
    
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
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    images = images.transpose(0, 1)
    
    return images, labels

def plotting_data(network, train_dataset, test_dataset, train_loader, test_loader, recurrence, device, run, k1):
    
    print("Assembling test data for 2D projection")
    ###
    with torch.no_grad():
        features, all_labs, all_decs, all_orig_ims = [], [], [], []
        for i,(data, labs) in enumerate(test_loader, 1):
            data = data.to(device)
            code_layer, decoded = network(data)
            features.append(to_np(code_layer.mean(0)))
            all_labs.append(labs)
            all_decs.append(to_np(decoded.squeeze().mean(0)))
            all_orig_ims.append(to_np(data.squeeze().mean(0)))
            
            del data, code_layer, decoded
            gc.collect()
            torch.cuda.empty_cache()
            
            if i % 1 == 0:
                print(f'-- {i}/{len(test_loader)} --')
    
        features = np.concatenate(features, axis = 0)
        all_labs = np.concatenate(all_labs, axis = 0)
        all_orig_ims = np.concatenate(all_orig_ims, axis = 0)
        all_decs = np.concatenate(all_decs, axis = 0)
    
        # del data, code_layer, decoded
        gc.collect()
        torch.cuda.empty_cache()
        
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
        
        #figsize=(12, 10)
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
        fig.savefig("figures/poisson_inputs.png", bbox_inches="tight", pad_inches=0)
        
        #figsize=(12, 10)
        fig, axs = plt.subplots(1, 10, layout="constrained")
        
        # Flatten the axis array for easier indexing
        axs = axs.flatten()    
        for i in range(10):
            axs[i].imshow(unique_ims[i][0], cmap='grey')
            axs[i].set(yticks = [], xticks=[])
            # if i==0:
            #     axs[i].set_ylabel(f"{run.name.replace("_", "/")}\n(Off)", rotation=0, labelpad=40)
        
        plt.tight_layout(pad=pad, h_pad=h, w_pad=w)
       
        fig.savefig(f"figures/result_summary_{run.name}.png", bbox_inches="tight", pad_inches=0)
        
        print("Plotting Spiking Input MNIST")

        # Plot originally input as image and as spiking representation - save gif.
        input_index = 1
        inputs, labels = next(iter(test_loader))
        inputs = inputs.to(device)
        img_spk_recs, img_spk_outs = network(inputs)
        new_inputs = inputs[:, input_index].squeeze().detach().cpu()
        labels = labels[input_index]
        
        new_img_spk_outs = img_spk_outs[:, input_index].squeeze().detach().cpu()
        new_img_spk_recs = img_spk_recs[:, input_index].squeeze().detach().cpu()
    
        del inputs, img_spk_recs, img_spk_outs
        
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"Plotting Spiking Input MNIST Animation - {labels}")
        fig, ax = plt.subplots()
        anim = splt.animator(new_inputs, fig, ax)
        HTML(anim.to_html5_video())
        anim.save(f"figures/spike_mnist_{labels}.gif")
        
        wandb.log({"Spike Animation": wandb.Video(f"figures/spike_mnist_{labels}.gif", fps=4, format="gif")}, commit = False)
        
        print("Plotting Spiking Output MNIST")
        t1, t2, t3 = 20, 50, 80
        fig =  plt.figure()
        ax1 = plt.subplot(1, 4, 1)
        ax1.imshow(new_inputs[t1], cmap='grey')
        plt.axis('off')
        plt.xlabel(f"t = {t1}")
        ax2 = plt.subplot(1, 4, 2)
        ax2.imshow(new_inputs[t2], cmap='grey')
        plt.axis('off')
        plt.xlabel(f"t = {t2}")
        ax3 = plt.subplot(1, 4, 3)
        ax3.imshow(new_inputs[t3], cmap='grey')
        plt.axis('off')
        plt.xlabel(f"t = {t3}")
        ax4 = plt.subplot(1, 4, 4)
        ax4.imshow(new_inputs.mean(axis=0), cmap='grey')
        plt.axis('off')
        plt.xlabel("Average over all timesteps")
        
        
        t0, t1, t2, t3 = 0, 25, 50, 75
        fig, ((ax1, ax2, ax3, ax4, ax5), (axa, axb, axc, axd, axe)) = plt.subplots(2, 5, layout="constrained", figsize=(10, 4))
    #    fontsize=8
    
        axa.imshow(new_img_spk_outs[t0:t0+5].mean(0), cmap="gray")
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        axa.set_xticks([])
        axa.set_yticks([])
    
        axb.imshow(new_img_spk_outs[t1:t1+5].mean(0), cmap="gray")
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True) # labels along the bottom edge are off
        axb.set_xticks([])
        axb.set_yticks([])
    
        axc.imshow(new_img_spk_outs[t2:t2+5].mean(0), cmap="gray")
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        axc.set_xticks([])
        axc.set_yticks([])
    
        axd.imshow(new_img_spk_outs[t3:t3+5].mean(0), cmap="gray")
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        axd.set_xticks([])
        axd.set_yticks([])
    
        axe.imshow(new_img_spk_outs.mean(axis=0), cmap="gray")
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        axe.set_xticks([])
        axe.set_yticks([])
    
        ax1.imshow(new_inputs[t0:t0+5].mean(0), cmap="gray")
        axa.set_xlabel(f"t = {t0}")
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True) # labels along the bottom edge are off
        ax1.set_xticks([])
        ax1.set_yticks([])
    
        ax2.imshow(new_inputs[t1:t1+5].mean(0), cmap="gray")
        axb.set_xlabel(f"t = {t1}")
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True) # labels along the bottom edge are off
        ax2.set_xticks([])
        ax2.set_yticks([])
    
        ax3.imshow(new_inputs[t2:t2+5].mean(0), cmap="gray")
        axc.set_xlabel(f"t = {t2}")
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True) # labels along the bottom edge are off
        ax3.set_xticks([])
        ax3.set_yticks([])
    
        ax4.imshow(new_inputs[t3:t3+5].mean(0), cmap="gray")
        axd.set_xlabel(f"t = {t3}")
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True) # labels along the bottom edge are off
        ax4.set_xticks([])
        ax4.set_yticks([])
    
    
        ax5.imshow(new_inputs.mean(axis=0), cmap="gray")
    
        axe.set_xlabel("Time Averaged")
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True) # labels along the bottom edge are off
        ax5.set_xticks([])
        ax5.set_yticks([])
    
        plt.tight_layout(pad=0, h_pad=0.7, w_pad=0)
        fig.savefig(f"figures/time_slices_{run.name}.png")
        
        print(f"Plotting Spiking Output MNIST Animation - {labels}")
        fig1, ax1 = plt.subplots()
        animrec = splt.animator(new_img_spk_outs, fig1, ax1)
        HTML(animrec.to_html5_video())
        animrec.save(f"figures/spike_mnistrec_{labels}.gif")
        in_size = train_dataset[0][0].shape[-1]
        num_rec = new_img_spk_recs.shape[-1]
        print("Rasters")
        dot_size = 0.5
        num_pixels = in_size**2
        round_pixels = int(ceil(num_pixels / 100.0)) * 100
        print(round_pixels)
        fig = plt.figure(facecolor="w", figsize=(10, 3))
        ax1 = plt.subplot(3, 1, 1)
        # rotation=0
        splt.raster(new_inputs.reshape(new_inputs.shape[0], -1), ax1, s=dot_size, c="black")
        plt.ylabel(ylabel=r"Input", labelpad=1)
        ax2 = plt.subplot(3, 1, 2)
        splt.raster(new_img_spk_recs.reshape(new_inputs.shape[0], -1), ax2, s=dot_size, c="black")
        plt.ylabel(ylabel=r"Code", labelpad=1)
        ax3 = plt.subplot(3, 1, 3)
        splt.raster(new_img_spk_outs.reshape(new_inputs.shape[0], -1), ax3, s=dot_size, c="black")
        plt.ylabel(ylabel=r"Output", labelpad=1)
    
        ax1.yaxis.set_label_coords(-0.01,0.501)
        ax2.yaxis.set_label_coords(-0.01,0.501)
        ax3.yaxis.set_label_coords(-0.01,0.501)
        ax1.set(xlim=[0, new_inputs.shape[0]], ylim=[-50, round_pixels+50], yticks = [], xticks=[])
        ax2.set(xlim=[0, new_inputs.shape[0]], ylim=[0-ceil(num_rec*0.1), round(num_rec*1.1)], xticks=[], yticks = [])
        ax3.set(xlim=[0, new_inputs.shape[0]], ylim=[-50, round_pixels+50], yticks = [], xlabel="Time, ms")

        fig.tight_layout()
        
        fig.savefig("figures/rasters.png")
        
        umap_file, sil_score, db_score, ax = umap_plt("./datafiles/"+run.name+".csv")
        
        if recurrence == True:
            fig = plt.figure(facecolor="w", figsize=(10, 5))
            ax1 = plt.subplot(1,2,1)
            weight_map(network.rlif_rec.recurrent.weight)
            plt.title("Posterior Weight Heatmap")
            ax2 = plt.subplot(1,2,2)
            sns.histplot(to_np(torch.flatten(network.rlif_rec.recurrent.weight)))
            plt.title("Posterior Weight Distribution")
            fig.savefig(f"figures/weightmap_{run.name}.png")
            plt.show()  
    
    
    del features, all_labs, all_decs, all_orig_ims, tsne, network, \
         input_index, new_inputs, \
         seen_labels, unique_ims, orig_ims, fig, axs, \
        t1, t2, t3, ax1, ax2, ax3, ax4, ax5, axa, axb, axc, axd, axe, fig1, animrec, \
        in_size, num_rec, dot_size, num_pixels, round_pixels
        
    gc.collect()
    torch.cuda.empty_cache()
    
    
    return labels, umap_file, sil_score, db_score

def print_params(network):
    print("\n--------------------------------------------------------")
    for name, param in network.named_parameters():
        if param.grad == None:
            print(f"{name} --> {param.shape} --> {param.requires_grad}")
        else:
            print(f"{name} --> {param.shape} --> {param.requires_grad}")
    print("--------------------------------------------------------\n")
    
def visTensor(network, ch=0, allkernels=False, nrow=12, padding=1):
    tensor = network.conv1.weight.clone().cpu()
    n,c,h,w = tensor.shape # i.e. [64, 12, 5, 5] --> 
    # print(f"Printing convolutional tensor of shape {tensor.shape} with grad = {tensor.requires_grad}")
    if allkernels:
        tensor = tensor.view(n*c, -1, w, h)
    elif c != 3:
        tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
    
    # print(f"Printing convolutional tensor of shape {tensor.shape} with grad = {tensor.requires_grad}")
    # print(f"Tensor: {tensor[0, 0, :, :]}")
    
    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utls.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    
    # print(f"Grid shape: {grid.shape}")
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    
    plt.axis('off')
    plt.ioff()
    plt.show()
    plt.savefig("figures/kernels.png")
    
def get_grad(network, mode="mean"):
    gradients = []
    for name, param in network.named_parameters():
        if param.grad is not None:
            if mode == "min":
                gradients.append(param.grad.detach().cpu().min())
            elif mode == "mean":
                gradients.append(param.grad.detach().cpu().mean())
            elif mode == "max":
                gradients.append(param.grad.detach().cpu().max())
            else:
                Exception("Enter Valid Mode")
    return np.mean(np.array(gradients))

def get_weights(network):
    weights = []
    for name, param in network.named_parameters():
        weights.append(param.detach().square().view(-1))
    
    return torch.cat(weights).sum()

def range_norm(tensor):
    return (tensor - tensor.min())/(tensor.max()-tensor.min())

class SignTransform:
    def __call__(self, sample):
        return torch.sign(sample)

def build_dvs_dataset(train_specs, input_specs = None, first_saccade = False):
    num_workers = train_specs["num_workers"]
    batch_size = train_specs["batch_size"]
    sensor_size = tonic.datasets.NMNIST.sensor_size
    subset_size = train_specs["subset_size"]
    
    persist = True if num_workers > 0 else False
    
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    new_size = (28, 28, 2)

    ds_transform = tonic.transforms.Compose([
                transforms.Denoise(filter_time=1000),
                tonic.transforms.Downsample(spatial_factor=0.21875, time_factor=0.001),
                transforms.ToFrame(sensor_size=new_size, n_time_bins=100),
                torch.from_numpy,
                dtype_transform(),
                SignTransform()
                ])

    # create datasets
    train_ds = tonic.datasets.DVSGesture("dataset/dvs_gesture", train=True, transform=ds_transform)
    test_ds = tonic.datasets.DVSGesture("dataset/dvs_gesture", train=False, transform=ds_transform)
    
    trainlen1 = len(train_ds)
    testlen1 = len(test_ds)
     
    train_dataset = create_subset(train_ds, int(len(train_ds)/subset_size))
    test_dataset = create_subset(test_ds, int(len(test_ds)/subset_size))
    
    trainlen2 = len(train_dataset)
    testlen2 = len(test_dataset)
    
    train_dataset = DiskCachedDataset(train_dataset, cache_path="C:/Users/lukea/Documents/Masters Project Code/MSc_Project/cache/fast_dataloading_trn")
    test_dataset = DiskCachedDataset(test_dataset, cache_path="C:/Users/lukea/Documents/Masters Project Code/MSc_Project/cache/fast_dataloading_test")

    print(f"Training: {trainlen1} -> {trainlen2}\nTesting: {testlen1} -> {testlen2}")
    print("\nMaking Dataloaders")

    # create dataloaders
    train_loader = DataLoader(train_dataset,
                              shuffle=True, 
                              batch_size=batch_size, 
                              collate_fn=tonic.collation.PadTensors(batch_first=False),
                              pin_memory=False, 
                              num_workers=num_workers, 
                              persistent_workers=persist)
    test_loader = DataLoader(test_dataset, 
                             shuffle=False, 
                             batch_size=batch_size, 
                             collate_fn=tonic.collation.PadTensors(batch_first=False), 
                             pin_memory=False, 
                             num_workers=num_workers, 
                             persistent_workers=persist)
    
    print("Finished Building Dataset")
    return train_dataset, train_loader, test_dataset, test_loader

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def get_test_accuracy(test_loader, net, device):
    total = 0
    correct = 0
   
    with torch.no_grad():
      net.eval()
      for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)
        
        # forward pass
        test_spk, _ = net(data.view(data.size(0), -1))
    
        # calculate total accuracy
        _, predicted = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    print(f"Total correctly classified test set images: {correct}/{total}")
    print(f"Test Set Accuracy: {100 * correct / total:.2f}%")
    return 100 * correct / total