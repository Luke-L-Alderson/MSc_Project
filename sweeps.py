# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:30:19 2024

@author: lukea
"""
import wandb

if __name__ == "__main__":
    lrs = [1e-3, 1e-4, 1e-5, 1e-6]
    bss = [64]
    
    sweep_config = {
        'program': "main.py",
        'method': 'grid',
        'metric': {'name': 'final_test_loss',
                   'goal': 'minimize'   
                   },
        'parameters': {'bs': {'values': bss},
                       'lr': {'values': lrs},
                       'epochs': {'value': 3}
                       }
        }
    
    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")



        
