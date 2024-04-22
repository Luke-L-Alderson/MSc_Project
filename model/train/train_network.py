from ..aux.functions import get_poisson_inputs, process_labels, mse_count_loss

import torch
import wandb

def train_network(network, train_loader, test_loader, input_specs, label_specs, train_specs, reporting = False):
    device = train_specs["device"]
    num_epochs = train_specs["num_epochs"]
    early_stop = train_specs["early_stop"]
    logging_freq = 10
    
    loss_fn = mse_count_loss(lambda_rate=train_specs["lambda_rate"],lambda_weights=None)
    optimizer = torch.optim.Adam(network.parameters(), lr=train_specs["lr"], betas=(0.9, 0.999))
    
    epoch_training_loss = [];
    epoch_testing_loss = [];
    
    for epoch in range(1, num_epochs+1):  # loop over the dataset multiple times
        print("5/5: Training network...")
        train_running_loss = 0.0
        test_running_loss = 0.0
        for i, (train_inputs, train_labels) in enumerate(train_loader, 1):
            if i == early_stop+1:
                break
                
            train_inputs = get_poisson_inputs(train_inputs, **input_specs).to(device)
            train_labels = process_labels(train_labels, **label_specs).to(device).type(torch.cuda.FloatTensor)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            train_spk_recs, train_spk_outs  = network(train_inputs)
            train_loss = loss_fn(train_spk_recs, train_spk_outs, train_inputs)
            
            if reporting:
                print(f'train_spk_recs: {train_spk_recs.size()}\
                      \ntrain_spk_outs: {train_spk_outs.size()}\
                      \ntrain_inputs: {train_inputs.size()}')
                
            train_loss.backward()
            optimizer.step()

            # print statistics
            train_running_loss += train_loss.item()
            
            if i % logging_freq == 0:    # print every 10 mini-batches
                print(f'[{epoch}/{num_epochs}, {i}/{len(train_loader)}] train_loss: {train_running_loss/logging_freq}')
                epoch_training_loss.append(train_running_loss/logging_freq)
                train_running_loss = 0.0
                
        print("5/5: Testing network...")
        for j, (test_inputs, test_labels) in enumerate(test_loader, 1):
            if j == early_stop+1:
                break
            test_inputs = get_poisson_inputs(test_inputs, **input_specs).to(device)
            test_labels = process_labels(test_labels, **label_specs).to(device).type(torch.cuda.FloatTensor)

            test_spk_recs, test_spk_outs  = network(test_inputs)
            test_loss = loss_fn(test_spk_recs, test_spk_outs, test_inputs)
            test_running_loss += test_loss.item()
            
            if j % logging_freq == 0:
                epoch_testing_loss.append(test_running_loss/logging_freq)
                test_running_loss = 0
            
            if reporting:
                print(f'train_spk_recs: {test_spk_recs.size()}\
                      \ntrain_spk_outs: {test_spk_outs.size()}\
                      \ntrain_inputs: {test_inputs.size()}')
                      
        wandb.log({"Training Loss": train_running_loss, "Testing Loss": test_running_loss})
        print(f"test_loss: {test_running_loss}")

    print('Training and Testing Finished')
    return network, epoch_training_loss, epoch_testing_loss