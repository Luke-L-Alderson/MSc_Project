from ..aux.functions import get_poisson_inputs, process_labels, mse_count_loss

import torch
import numpy as np
import wandb
from math import floor, ceil
from datetime import datetime

def train_network(network, train_loader, test_loader, input_specs, label_specs, train_specs):
    startTime = datetime.now()
    print(startTime)
    device = train_specs["device"]
    num_epochs = train_specs["num_epochs"]
    early_stop = train_specs["early_stop"]
    batch_size = train_specs["batch_size"]
    train_logging_freq = ceil(0.05*len(train_loader))
    test_logging_freq = ceil(0.05*len(test_loader))
    
    loss_fn = mse_count_loss(lambda_rate=train_specs["lambda_rate"],lambda_weights=None)
    optimizer = torch.optim.Adam(network.parameters(), lr=train_specs["lr"], betas=(0.9, 0.999))
    
    epoch_training_loss = [];
    epoch_testing_loss = [];
    
    for epoch in range(1, num_epochs+1):
        print("\nTraining!")
        train_running_loss = 0.0 
        test_running_loss = 0.0

        for i, (train_inputs, train_labels) in enumerate(train_loader, 1): #something here takes 30 seconds
            if i == 1: print(datetime.now() - startTime)     
            if i == early_stop+1:
                print("Stopped early")
                break
   
            train_inputs = get_poisson_inputs(train_inputs, **input_specs).to(device)

            optimizer.zero_grad()

            train_spk_recs, train_spk_outs  = network(train_inputs) #takes a long time

            train_loss = loss_fn(train_spk_recs, train_spk_outs, train_inputs)
      
            train_loss.backward()

            optimizer.step()

            train_running_loss += train_loss.detach()#.item()

            if i % train_logging_freq == 0:
                print(f'[{epoch}/{num_epochs}, {i}/{len(train_loader)}] Training Loss: {train_running_loss/train_logging_freq}')
                epoch_training_loss.append(train_running_loss/train_logging_freq)
                wandb.log({"Training Loss": epoch_training_loss[-1]})
                train_running_loss = 0.0
                
        print("\nTesting!")
        with torch.no_grad():
            for j, (test_inputs, test_labels) in enumerate(test_loader):
                if j == early_stop:
                    print("Stopped early")
                    break
                
                test_inputs = get_poisson_inputs(test_inputs, **input_specs).to(device)
                test_spk_recs, test_spk_outs  = network(test_inputs)
                test_loss = loss_fn(test_spk_recs, test_spk_outs, test_inputs)
                test_running_loss += test_loss.detach()#.item()
                
                
                if j % test_logging_freq == 0:
                    print(f'[{epoch}/{num_epochs}, {j+1}/{len(test_loader)}]')
                    epoch_testing_loss.append(test_running_loss/test_logging_freq)
                    wandb.log({"Testing Loss": epoch_testing_loss[-1]})
                    test_running_loss = 0
  
            print(f"Testing Loss: {epoch_testing_loss[-1]}")

    print('\nTraining and Testing Finished')
    return network, epoch_training_loss, epoch_testing_loss, epoch_training_loss[-1].item(), epoch_testing_loss[-1].item()#, \
        #features, all_labs, all_decs, all_orig_ims