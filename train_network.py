from helpers import nmse_count_loss, get_grad
import torch
import wandb
from datetime import datetime
from math import ceil
from snntorch import utils
import gc
from torch import nn

def train_network(network, train_loader, test_loader, train_specs, k1=25, state="present"):
    startTime = datetime.now()
    device = train_specs["device"]
    num_epochs = train_specs["num_epochs"]
    train_logging_freq = ceil(0.1*len(train_loader))
    test_logging_freq = ceil(0.1*len(test_loader))
    loss_fn = nmse_count_loss(lambda_rate = train_specs["lambda_rate"], ntype=train_specs["norm_type"])
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
            #print(train_spk_recs.sum())
            train_loss = loss_fn(train_spk_outs, train_inputs, train_spk_recs)/CV
            train_loss.backward()
            wandb.log({"Min Gradients": get_grad(network, mode="min"),
                        "Mean Gradients": get_grad(network, mode="mean"),
                        "Max Gradients": get_grad(network, mode="max")
                        }) 
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

def train_network_bptt(network, train_loader, test_loader, train_specs, k1=25, state="present"):
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
            #utils.reset(network)
            optimizer.zero_grad()
            iterTime = datetime.now()   
            train_inputs = train_inputs.to(device)  # Input shape: [ts, bs, ch, h, w]
            num_steps_trn = train_inputs.shape[0]
            train_spk_recs, train_spk_outs  = network(train_inputs)
            
            
            # BPTT(n , n) - BPTT for whole sequence followed by whole backprop
            # BPTT(1, n) - BPTT for 1 timestep, followed by whole backprop
            # BPTT(k1, k2) - BPTT for k1 timesteps, follwed by a backprop over k2 timesteps
            total_loss = 0
            past_t = 0
            
            for t in range(0, num_steps_trn, k1):
                CV = train_inputs[t:t+k1].sum(0).std().item()/train_inputs[t:t+k1].sum(0).mean().item()
                
                if state == "past":
                    train_loss = loss_fn(train_spk_outs[past_t:past_t+k1], train_inputs[t:t+k1])/CV
                elif state == "present":
                    train_loss = loss_fn(train_spk_outs[t:t+k1], train_inputs[t:t+k1])/CV
                elif state == "future":
                    train_loss = loss_fn(train_spk_outs[t+k1:t+2*k1], train_inputs[t:t+k1])/CV
                else:
                    Exception("Error")
                
                total_loss += train_loss
                train_running_loss += train_loss.item()
                past_t = t
            
            
            total_loss.backward()
            wandb.log({"Min Gradients": get_grad(network, mode="min"),
                        "Mean Gradients": get_grad(network, mode="mean"),
                        "Max Gradients": get_grad(network, mode="max")
                        }) 
            
            optimizer.step()
            optimizer.zero_grad()
            utils.reset(network)
                       
            if i % train_logging_freq == 0: #:.4f
                print(f'[{epoch}/{num_epochs}, {i}/{len(train_loader)}] Training Loss: {train_running_loss/(train_logging_freq*num_steps_trn/k1)} - Iteration Time: {datetime.now()-iterTime} - Data Size: {train_inputs.shape}-->{state}')
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
                CV = test_inputs.sum(0).std().item()/test_inputs.sum(0).mean().item()
                test_loss = loss_fn(test_spk_outs, test_inputs)/CV
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


def train_network_c(network, train_loader, test_loader, train_specs, k1=25, state="present"):
    startTime = datetime.now()
    device = train_specs["device"]
    num_epochs = train_specs["num_epochs"]
    train_logging_freq = ceil(0.1*len(train_loader))
    test_logging_freq = ceil(0.1*len(test_loader))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=train_specs["lr"], betas=(0.9, 0.999))
    stds = []
    means = []
    epoch_training_loss = [];
    epoch_testing_loss = [];
    total, correct = 0, 0
    for epoch in range(1, num_epochs+1):
        epochTime = datetime.now() 
        print(f"\nTraining - {datetime.now()}")
        train_running_loss = 0.0 
        test_running_loss = 0.0

        for i, (train_inputs, train_labels) in enumerate(train_loader, 1):
            
            iterTime = datetime.now()    
            optimizer.zero_grad()
            utils.reset(network)
            means.append(train_inputs.sum(0).mean().item())
            stds.append(train_inputs.sum(0).std().item())
            CV = train_inputs.sum(0).std().item()/train_inputs.sum(0).mean().item()
                
            num_steps = train_inputs.shape[0]    
            train_inputs, train_labels = train_inputs.to(device), train_labels.type(torch.LongTensor).to(device)
            
            spikes, mems  = network(train_inputs)
            
            train_loss = torch.zeros(1, device=device)

            for step in range(num_steps):
                train_loss += loss_fn(mems[step], train_labels)#/CV
            
            train_loss.backward()
            wandb.log({"Min Gradients": get_grad(network, mode="min"),
                        "Mean Gradients": get_grad(network, mode="mean"),
                        "Max Gradients": get_grad(network, mode="max")
                        }) 
            optimizer.step()
            
            train_running_loss += train_loss.item()
            del spikes, mems, train_inputs
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
                test_inputs, test_labels = test_inputs.to(device), test_labels.type(torch.LongTensor).to(device)
                CV = test_inputs.sum(0).std().item()/test_inputs.sum(0).mean().item()
                spikes, mems  = network(test_inputs)
                print(spikes.shape)
                test_loss = torch.zeros(1, device=device)
                
    
                for step in range(num_steps):
                    test_loss += loss_fn(mems[step], test_labels)#/CV
                
                test_running_loss += test_loss.item()
                
                # calculate total accuracy
                _, predicted = spikes.sum(0).max(-1)
                
                #print(mems.shape, predicted.shape, test_labels.shape)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()
                # print(predicted == test_labels)
                # print((predicted == test_labels).sum().item())
                # print(correct)
                del spikes, mems, test_inputs
                
                if j % test_logging_freq == 0:
                    print(f'[{epoch}/{num_epochs}, {j}/{len(test_loader)}] - Label Size: {test_labels.shape}')
                    epoch_testing_loss.append(test_running_loss/test_logging_freq)
                    test_running_loss = 0
            
            accuracy = 100*correct/total
            wandb.log({"Accuracy": accuracy})
            
        print(f'Testing Loss: {epoch_testing_loss[-1]:.4f} - Epoch Time: {datetime.now()-epochTime}')
        
        try:
            wandb.log({"Testing Loss": epoch_testing_loss[-1]})
        except:
            pass
    
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f'\nTraining and Testing Finished - Time: {datetime.now() - startTime} - Mean: {np.array(means).mean()}, Std: {np.array(stds).mean()}')
    return network, epoch_training_loss, epoch_testing_loss, epoch_training_loss[-1], epoch_testing_loss[-1]