from helpers import get_poisson_inputs, rmse_count_loss, nmse_count_loss, print_params, get_grad
import torch
import wandb
from brian2 import *
from datetime import datetime
import torch.nn.functional as F
from math import ceil
from torch.profiler import profile, record_function, ProfilerActivity
from snntorch import utils
from snntorch.functional import mse_count_loss
from torch.nn.utils import clip_grad_norm_
from matplotlib import pyplot as plt
import gc
def train_network(network, train_loader, test_loader, train_specs, k1=25):
    startTime = datetime.now()
    device = train_specs["device"]
    num_epochs = train_specs["num_epochs"]
    train_logging_freq = ceil(0.1*len(train_loader))
    test_logging_freq = ceil(0.1*len(test_loader))
    loss_fn = nmse_count_loss(ntype=train_specs["norm_type"])
    optimizer = torch.optim.Adam(network.parameters(), lr=train_specs["lr"], betas=(0.9, 0.999))
    stds = []
    means = []
    epoch_training_loss = [];
    epoch_testing_loss = [];
    
    for epoch in range(1, num_epochs+1):
        epochTime = datetime.now() 
        print(f"\nTraining - {datetime.now()}")
        train_running_loss = 0.0 
        test_running_loss = 0.0

        for i, (train_inputs, train_labels) in enumerate(train_loader, 1):
            iterTime = datetime.now()    
            optimizer.zero_grad()
            means.append(train_inputs.sum(0).mean().item())
            stds.append(train_inputs.sum(0).std().item())
            CV = train_inputs.sum(0).std().item()/train_inputs.sum(0).mean().item()
                
                
            train_inputs = train_inputs.to(device)
            train_spk_recs, train_spk_outs  = network(train_inputs)         
            train_loss = loss_fn(train_spk_outs, train_inputs, train_spk_recs)/CV
            train_loss.backward()
            optimizer.step()
            train_running_loss += train_loss.item()
            del train_spk_recs, train_spk_outs, train_inputs
            if i % train_logging_freq == 0:
                print(f'[{epoch}/{num_epochs}, {i}/{len(train_loader)}] Training Loss: {train_running_loss/train_logging_freq:.4f} - Iteration Time: {datetime.now()-iterTime}')
                epoch_training_loss.append(train_running_loss/train_logging_freq)
                try:
                    wandb.log({"Training Loss": epoch_training_loss[-1]})
                except:
                    pass
                train_running_loss = 0.0
            
                
        print(f"\nTesting - {datetime.now()}")
        with torch.no_grad():
            for j, (test_inputs, test_labels) in enumerate(test_loader, 1):
                test_inputs = test_inputs.to(device)
                CV = test_inputs.sum(0).std().item()/test_inputs.sum(0).mean().item()
                test_spk_recs, test_spk_outs  = network(test_inputs)
                test_loss = loss_fn(test_spk_outs, test_inputs, test_spk_recs)/CV
                test_running_loss += test_loss.item()
                del test_spk_recs, test_spk_outs, test_inputs
                if j % test_logging_freq == 0:
                    print(f'[{epoch}/{num_epochs}, {j}/{len(test_loader)}]')
                    epoch_testing_loss.append(test_running_loss/test_logging_freq)
                    test_running_loss = 0

        print(f'Testing Loss: {epoch_testing_loss[-1]:.4f} - Epoch Time: {datetime.now()-epochTime}')
        
        try:
            wandb.log({"Testing Loss": epoch_testing_loss[-1]})
        except:
            pass
    
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f'\nTraining and Testing Finished - Time: {datetime.now() - startTime} - Mean: {np.array(means).mean()}, Std: {np.array(stds).mean()}')
    return network, epoch_training_loss, epoch_testing_loss, epoch_training_loss[-1], epoch_testing_loss[-1]
    
def train_network_ns(network, train_loader, test_loader, train_specs):
    startTime = datetime.now()
    print(startTime)
    device = train_specs["device"]
    num_epochs = train_specs["num_epochs"]
    train_logging_freq = ceil(0.1*len(train_loader))
    test_logging_freq = ceil(0.1*len(test_loader))
    loss_fn = nmse_count_loss(ntype=train_specs["norm_type"])
    stds = []
    means = []
    optimizer = torch.optim.Adam(network.parameters(), lr=train_specs["lr"], betas=(0.9, 0.999))
    
    epoch_training_loss = [];
    epoch_testing_loss = [];
    
    for epoch in range(1, num_epochs+1):
        epochTime = datetime.now() 
        print(f"\nTraining - {datetime.now()}")
        train_running_loss = 0.0 
        test_running_loss = 0.0

        for i, (train_inputs, train_labels) in enumerate(train_loader, 1):
            iterTime = datetime.now()
            means.append(train_inputs.mean().item())
            stds.append(train_inputs.std().item())
            CV = train_inputs.std().item()/train_inputs.mean().item()
            
            train_inputs = train_inputs.to(device)
            train_latent, train_outputs = network(train_inputs)
            
            train_loss = loss_fn(train_outputs, train_inputs)/CV
            train_loss.backward()
            optimizer.step()
            train_running_loss += train_loss.item()
            optimizer.zero_grad()
            
            if i % train_logging_freq == 0:
                print(f'[{epoch}/{num_epochs}, {i}/{len(train_loader)}] Training Loss: {train_running_loss/train_logging_freq:.4f} - Iteration Time: {datetime.now()-iterTime}')
                epoch_training_loss.append(train_running_loss/train_logging_freq)
                wandb.log({"Training Loss": epoch_training_loss[-1],
                           "Min Gradients": get_grad(network, mode="min"),
                           "Mean Gradients": get_grad(network, mode="mean"),
                           "Max Gradients": get_grad(network, mode="max")
                           }) 
                train_running_loss = 0.0
                
        print(f"\nTesting - {datetime.now()}")
        with torch.no_grad():
            for j, (test_inputs, test_labels) in enumerate(test_loader, 1):
                test_inputs = test_inputs.to(device)
                train_latent,test_outputs  = network(test_inputs)
                CV = test_inputs.std().item()/test_inputs.mean().item()
                test_loss = loss_fn(test_outputs, test_inputs)/CV
                test_running_loss += test_loss.item()
                
                if j % test_logging_freq == 0:
                    print(f'[{epoch}/{num_epochs}, {j}/{len(test_loader)}]')
                    epoch_testing_loss.append(test_running_loss/test_logging_freq)
                    test_running_loss = 0

        print(f'Testing Loss: {epoch_testing_loss[-1]:.4f} - Epoch Time: {datetime.now()-epochTime}')
        wandb.log({"Testing Loss": epoch_testing_loss[-1]})

    print(f'\nTraining and Testing Finished - Time: {datetime.now() - startTime}')
    return network, epoch_training_loss, epoch_testing_loss, epoch_training_loss[-1], epoch_testing_loss[-1]

def train_network_bptt(network, train_loader, test_loader, train_specs, k1=25):
    startTime = datetime.now()
    print(startTime)
    device = train_specs["device"]
    num_epochs = train_specs["num_epochs"]
    train_logging_freq = ceil(0.1*len(train_loader))
    test_logging_freq = ceil(0.1*len(test_loader))
    loss_fn = nmse_count_loss(ntype=train_specs["norm_type"], lambda_rate=0)
    optimizer = torch.optim.Adam(network.parameters(), lr=train_specs["lr"], betas=(0.9, 0.999))
    k1 = k1
    #k2 = 25
    epoch_training_loss = [];
    epoch_testing_loss = [];
    
    for epoch in range(1, num_epochs+1):
        epochTime = datetime.now() 
        print(f"\nTraining - {datetime.now()}")
        train_running_loss = 0.0 
        test_running_loss = 0.0

        for i, (train_inputs, train_labels) in enumerate(train_loader, 1):
            # Detach hidden state to prevent backpropagating through the entire history
            utils.reset(network)
            optimizer.zero_grad()
            iterTime = datetime.now()   
            train_inputs = train_inputs.to(device)  # Input shape: [ts, bs, ch, h, w]
            num_steps_trn = train_inputs.shape[0]
                        
            # BPTT(n , n) - BPTT for whole sequence followed by whole backprop
            # BPTT(1, n) - BPTT for 1 timestep, followed by whole backprop
            # BPTT(k1, k2) - BPTT for k1 timesteps, follwed by a backprop over k2 timesteps
            total_loss = 0       
            for t in range(0, num_steps_trn, k1):
                train_inputs_k1 = train_inputs[t:t+k1] # split input into chunks of k1
                train_spk_recs, train_spk_outs  = network(train_inputs_k1) # Forward pass through k1 time steps
                
                #if t >= k1:
                train_loss = loss_fn(train_spk_outs, train_inputs_k1)
                train_running_loss += train_loss.item()
                
                train_loss.backward()
                wandb.log({"Min Gradients": get_grad(network, mode="min"),
                            "Mean Gradients": get_grad(network, mode="mean"),
                            "Max Gradients": get_grad(network, mode="max")
                            }) 
                
                optimizer.step()
                optimizer.zero_grad()
                utils.reset(network)
                       
            if i % train_logging_freq == 0: #:.4f
                print(f'[{epoch}/{num_epochs}, {i}/{len(train_loader)}] Training Loss: {train_running_loss/(train_logging_freq*num_steps_trn/k1)} - Iteration Time: {datetime.now()-iterTime} - Data Size: {train_inputs.shape}-->{train_inputs_k1.shape}')
                epoch_training_loss.append(train_running_loss/(train_logging_freq*num_steps_trn/k1))
                try:
                    wandb.log({"Training Loss": epoch_training_loss[-1]})
                except:
                    pass
                train_running_loss = 0.0
                
        print(f"\nTesting - {datetime.now()}")
        with torch.no_grad():
            for j, (test_inputs, test_labels) in enumerate(test_loader, 1):
                test_inputs = test_inputs.to(device)
                
                # Forward pass through k1 time steps
                test_spk_recs, test_spk_outs  = network(test_inputs)
                
                test_loss = loss_fn(test_spk_outs, test_inputs)
                test_running_loss += test_loss.item()
                
                if j % test_logging_freq == 0:
                    print(f'[{epoch}/{num_epochs}, {j}/{len(test_loader)}]')
                    epoch_testing_loss.append(test_running_loss/test_logging_freq)
                    test_running_loss = 0

        print(f'Testing Loss: {epoch_testing_loss[-1]:.4f} - Epoch Time: {datetime.now()-epochTime}')
        
        try:
            wandb.log({"Testing Loss": epoch_testing_loss[-1]})
        except:
            pass
    
    print(f'\nTraining and Testing Finished - Time: {datetime.now() - startTime}')
    return network, epoch_training_loss, epoch_testing_loss, epoch_training_loss[-1], epoch_testing_loss[-1]

'''
alal_loss = 0
    for t in range(0, seq_length, k1):
        # Truncate the sequence
        x_k1 = data[t:t+k1]
        y_k1 = targets[t:t+k1]
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs, hidden = model(x_k1, hidden)
        
        # Compute loss
        loss = criterion(outputs, y_k1)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Detach hidden state to prevent backpropagating through the entire history
        utils.reset(network)
        
        total_loss += loss.item()
'''