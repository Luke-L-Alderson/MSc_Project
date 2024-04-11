import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import utils as utls

import snntorch as snn
import snntorch.spikeplot as splt
from snntorch import utils
from snntorch import surrogate

from model.train.train_network import train_network
from model.image_to_latent import Net
from model.image_to_image import SAE
from model.aux.functions import get_poisson_inputs, process_labels, mse_count_loss
from data.aux.dataset import H5Dataset
from data_generator.ds_generator import make_dataset, make_exam_tests
from testing.exam import get_exam_per_constant

def train(network, trainloader, opti, epoch):

    network=network.train()
    train_loss_hist=[]
    for batch_idx, (real_img, labels) in enumerate(trainloader):
        opti.zero_grad()
        real_img = real_img.to(device)
        labels = labels.to(device)

        #Pass data into network, and return reconstructed image from Membrane Potential at t = -1
        x_recon = network(real_img) #Dimensions passed in: [Batch_size,Input_size,Image_Width,Image_Length]

        #Calculate loss
        loss_val = F.mse_loss(x_recon, real_img)

        print(f'Train[{epoch}/{max_epoch}][{batch_idx}/{len(trainloader)}] Loss: {loss_val.item()}')

        loss_val.backward()
        opti.step()

        #Save reconstructed images every at the end of the epoch
        if batch_idx == len(trainloader)-1:
            # NOTE: you need to create training/ and testing/ folders in your chosen path
            utls.save_image((real_img+1)/2, f'figures/training/epoch{epoch}_finalbatch_inputs.png')
            utls.save_image((x_recon+1)/2, f'figures/training/epoch{epoch}_finalbatch_recon.png')
    return loss_val

def test(network, testloader, opti, epoch):
    network=network.eval()
    test_loss_hist=[]
    with torch.no_grad(): #no gradient this time
        for batch_idx, (real_img, labels) in enumerate(testloader):
            real_img = real_img.to(device)#
            labels = labels.to(device)
            x_recon = network(real_img)

            loss_val = F.mse_loss(x_recon, real_img)

            print(f'Test[{epoch}/{max_epoch}][{batch_idx}/{len(testloader)}]  Loss: {loss_val.item()}')#, RECONS: {recons_meter.avg}, DISTANCE: {dist_meter.avg}')

            if batch_idx == len(testloader)-1:
                utls.save_image((real_img+1)/2, f'figures/testing/epoch{epoch}_finalbatch_inputs.png')
                utls.save_image((x_recon+1)/2, f'figures/testing/epoch{epoch}_finalbatch_recons.png')
    return loss_val


