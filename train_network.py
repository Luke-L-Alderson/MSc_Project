from helpers import get_poisson_inputs, rmse_count_loss, nmse_count_loss
import torch
import wandb
from brian2 import *
from datetime import datetime
import torch.nn.functional as F
from math import ceil
from torch.profiler import profile, record_function, ProfilerActivity
from snntorch import utils

def train_network(network, train_loader, test_loader, train_specs):
    startTime = datetime.now()
    device = train_specs["device"]
    num_epochs = train_specs["num_epochs"]
    train_logging_freq = ceil(0.1*len(train_loader))
    test_logging_freq = ceil(0.1*len(test_loader))
    loss_fn = nmse_count_loss(ntype=train_specs["norm_type"])
    
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
            train_inputs = train_inputs.to(device)
            train_spk_recs, train_spk_outs  = network(train_inputs)         
            train_loss = loss_fn(train_spk_outs, train_inputs, train_spk_recs)
            train_loss.backward()
            optimizer.step()
            train_running_loss += train_loss.item()
            optimizer.zero_grad()
            
            if i % train_logging_freq == 0:
                print(f'[{epoch}/{num_epochs}, {i}/{len(train_loader)}] Training Loss: {train_running_loss/train_logging_freq:.4f} - Iteration Time: {datetime.now()-iterTime} - Data Size: {train_inputs.shape}')
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
                test_spk_recs, test_spk_outs  = network(test_inputs)
                test_loss = loss_fn(test_spk_outs, test_inputs, test_spk_recs)
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
    
def train_network_ns(network, train_loader, test_loader, train_specs):
    startTime = datetime.now()
    print(startTime)
    device = train_specs["device"]
    num_epochs = train_specs["num_epochs"]
    train_logging_freq = ceil(0.1*len(train_loader))
    test_logging_freq = ceil(0.1*len(test_loader))
    loss_fn = nmse_count_loss(ntype=train_specs["norm_type"])
    
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
            train_inputs = train_inputs.to(device)
            train_latent, train_outputs  = network(train_inputs)
            
            
            train_loss = loss_fn(train_outputs, train_inputs)
            train_loss.backward()
            optimizer.step()
            train_running_loss += train_loss.item()
            optimizer.zero_grad()
            
            if i % train_logging_freq == 0:
                print(f'[{epoch}/{num_epochs}, {i}/{len(train_loader)}] Training Loss: {train_running_loss/train_logging_freq:.4f} - Iteration Time: {datetime.now()-iterTime}')
                epoch_training_loss.append(train_running_loss/train_logging_freq)
                wandb.log({"Training Loss": epoch_training_loss[-1]})
                train_running_loss = 0.0
                
        print(f"\nTesting - {datetime.now()}")
        with torch.no_grad():
            for j, (test_inputs, test_labels) in enumerate(test_loader, 1):
                test_inputs = test_inputs.to(device)
                train_latent,test_outputs  = network(test_inputs)
                test_loss = loss_fn(test_outputs, test_inputs)
                test_running_loss += test_loss.item()
                
                if j % test_logging_freq == 0:
                    print(f'[{epoch}/{num_epochs}, {j}/{len(test_loader)}]')
                    epoch_testing_loss.append(test_running_loss/test_logging_freq)
                    test_running_loss = 0

        print(f'Testing Loss: {epoch_testing_loss[-1]:.4f} - Epoch Time: {datetime.now()-epochTime}')
        wandb.log({"Testing Loss": epoch_testing_loss[-1]})

    print(f'\nTraining and Testing Finished - Time: {datetime.now() - startTime}')
    return network, epoch_training_loss, epoch_testing_loss, epoch_training_loss[-1], epoch_testing_loss[-1]

def train_network_bptt(network, train_loader, test_loader, train_specs):
    torch.autograd.set_detect_anomaly(True)
    startTime = datetime.now()
    print(startTime)
    device = train_specs["device"]
    num_epochs = train_specs["num_epochs"]
    train_logging_freq = ceil(0.1*len(train_loader))
    test_logging_freq = ceil(0.1*len(test_loader))
    #loss_fn = nmse_count_loss(ntype=train_specs["norm_type"])
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=train_specs["lr"], betas=(0.9, 0.999))
    
    epoch_training_loss = [];
    epoch_testing_loss = [];
    
    for epoch in range(1, num_epochs+1):
        epochTime = datetime.now() 
        print(f"\nTraining - {datetime.now()}")
        train_running_loss = 0.0 
        test_running_loss = 0.0

        for i, (train_inputs, train_labels) in enumerate(train_loader, 1):
            utils.reset(network)
            iterTime = datetime.now()   
            train_inputs = train_inputs.to(device)  # Input shape: [ts, bs, ch, h, w]
            num_steps_trn = train_inputs.shape[0]
            
            training_loss = torch.zeros((1), device=device)
            
            for t_trn in range(num_steps_trn): # time loop move outside the network
                train_spk_recs, train_spk_outs  = network(train_inputs[t_trn]) # In: [bs, ch, h, w] --> Out: [bs, ch, h, w]
                training_loss = training_loss + loss_fn(train_spk_outs, train_inputs[t_trn]) # loss_fn = torch.nn.MSELoss()
                training_loss.backward(retain_graph=True)
            
            optimizer.step()
            optimizer.zero_grad()
            
            if i % train_logging_freq == 0:
                print(f'[{epoch}/{num_epochs}, {i}/{len(train_loader)}] Training Loss: {train_running_loss/train_logging_freq:.4f} - Iteration Time: {datetime.now()-iterTime} - Data Size: {train_inputs.shape}')
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