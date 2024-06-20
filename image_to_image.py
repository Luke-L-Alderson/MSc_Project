import torch
import torch.nn as nn
import numpy as np
import snntorch as snn
from snntorch import surrogate
import torch.nn.functional as F
#from brian2 import *

class SAE(nn.Module):
    def __init__(self, tp, netp, fp, cp, device, recurrence):
        super().__init__()
        self.device = device
        self.recurrence = recurrence
        #save and proces param dicts
        beta, num_rec, num_latent, depth, num_conv2 = self.process_params(tp, netp, fp, cp)
        threshold = netp["v_th"]
        
        #define gradient
        spike_grad = surrogate.fast_sigmoid(slope=25)
          
        #convolution (encoder) - Input size: [8, 1, 28, 28]
        self.conv1 = nn.Conv2d(depth, cp["channels_1"], (cp["filter_1"], cp["filter_1"]))
        self.lif_conv1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold)
        # conv1 output size: [8, 12, 26, 26]
        self.conv2 = nn.Conv2d(cp["channels_1"], cp["channels_2"], (cp["filter_2"], cp["filter_2"]))
        self.lif_conv2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold)
        # conv2 output size: [8, 64, 24, 24]  

        # recurrent (latent)
        self.ff_in = nn.Linear(num_conv2, num_rec) # 64x24x24 (12288) -> 100
        self.ff_rec = nn.Linear(num_rec, num_rec)  # 100 -> 100 (same nodes)
        self.net_rec = snn.Leaky(beta=beta, spike_grad=spike_grad, reset_mechanism="zero", threshold = threshold)
        
        self.ff_out = nn.Linear(num_rec, num_conv2)


        # needs to be [8, 64, 24, 24] format for convTrans
        # convolution (decoder)
        
        self.deconv2 = nn.ConvTranspose2d(cp["channels_2"], cp["channels_1"], (cp["filter_2"], cp["filter_2"])) 
        self.lif_deconv2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold)
        # deconv2 output size: [8, 12, 26, 26]  
        
        self.reconstruction = nn.ConvTranspose2d(cp["channels_1"], depth, (cp["filter_1"], cp["filter_1"]))
        self.lif_reconstruction = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold)
        # recontruction should be [8, 3, 28, 28]
        
        
    def forward(self, x, recorded_vars=None):
        num_rec, num_conv2, noise_amp, channels_1, conv1_size, channels_2, conv2_size = self.network_params["num_rec"], self.network_params["num_conv2"], self.network_params["noise_amplitude"], self.convolution_params["channels_1"], self.convolution_params["conv1_size"], self.convolution_params["channels_2"], self.convolution_params["conv2_size"]

        try:
          x = x.type(torch.cuda.FloatTensor)
        except:
          x = x.type(torch.FloatTensor)
        
        # Initialize hidden states and outputs at t=0
        batch_size = len(x[0])       
        mem_conv1 = self.lif_conv1.init_leaky()
        mem_conv2 = self.lif_conv2.init_leaky()
        mem_rec = self.net_rec.init_leaky()
        mem_deconv2 = self.lif_deconv2.init_leaky()
        mem_reconstruction = self.lif_reconstruction.init_leaky()
        spk_conv1 = torch.zeros(batch_size, channels_1, conv1_size, conv1_size, device = self.device)
        spk_conv2 = torch.zeros(batch_size, channels_2, conv2_size, conv2_size, device = self.device)
        spk_deconv1 = torch.zeros(batch_size, channels_2, conv2_size, conv2_size, device = self.device)
        spk_deconv2 = torch.zeros(batch_size, channels_1, conv1_size, conv1_size, device = self.device)
        spk_rec = torch.zeros(batch_size, num_rec, device = self.device)
        spk_reconstruction = torch.zeros(batch_size, channels_2, conv2_size, conv2_size, device = self.device)
        
        spk_outs, spk_recs, mem_outs, mem_recs = [], [], [], []
        
        # Record latent and output layer
        if recorded_vars:
            recorded = {key: [] for key in recorded_vars}

        for timestep in range(self.time_params["num_timesteps"]):
            
            # convolution (encoder) - layer 1
            curr_conv1 = self.conv1(x[timestep])
            spk_conv1, mem_conv1 = self.lif_conv1(curr_conv1, mem_conv1)
            mem_conv1 += noise_amp*torch.randn(mem_conv1.shape, device = self.device)
            
            # convolution (encoder) - layer 2
            curr_conv2 = self.conv2(spk_conv1)
            spk_conv2, mem_conv2 = self.lif_conv2(curr_conv2, mem_conv2)
            mem_conv2 += noise_amp*torch.randn(mem_conv2.shape, device = self.device)
            
            #recurrent layer (encoder)
            curr_in = self.ff_in(spk_conv2.view(batch_size, -1))
            curr_rec = self.ff_rec(spk_rec)
            curr_total = curr_in + self.recurrence*curr_rec
            spk_rec, mem_rec = self.net_rec(curr_total, mem_rec) #can set curr_total to curr_in, param in front

            mem_rec += noise_amp*torch.randn(mem_rec.shape, device = self.device)
            
            curr_out = self.ff_out(spk_rec)
            
            # convolution (decoder) - undo layer 2
            curr_deconv2 = self.deconv2(curr_out.view(curr_out.size(0), channels_2, conv2_size, conv2_size))
            spk_deconv2, mem_deconv2 = self.lif_conv2(curr_deconv2, mem_deconv2)
            mem_deconv2 += noise_amp*torch.randn(mem_deconv2.shape, device = self.device)
            
            # convolution (decoder) - undo layer 1
            curr_reconstruction = self.reconstruction(spk_deconv2)
            spk_reconstruction, mem_reconstruction = self.lif_reconstruction(curr_reconstruction, mem_reconstruction)
            mem_reconstruction += noise_amp*torch.randn(mem_reconstruction.shape, device = self.device)
            
            ''' Noise if needed
            #mem_latent += noise_amp*torch.randn(mem_latent.shape).to(self.device)
            '''
            
            spk_recs.append(spk_rec)
            spk_outs.append(spk_reconstruction)
            #mem_recs.append(mem_rec)
            #mem_outs.append(mem_reconstruction)

            if recorded_vars:
                for key in recorded:
                    recorded[key].append(locals()[key])
        
        if recorded_vars:
            for key, item in recorded.items():
                recorded[key] = torch.stack(item)
            return recorded
        return torch.stack(spk_recs), torch.stack(spk_outs)
    
    
    def process_params(self, tp, netp, fp, cp):
        netp["beta"] = np.exp(-tp["dt"]/netp["tau_m"])
        netp["noise_amplitude"] = netp["eta"]*np.sqrt((1 - np.exp(-2*tp["dt"]/netp["tau_m"]))/2)
        tp["num_timesteps"] = int(tp["total_time"]/tp["dt"])
        
        cp["conv1_size"] = fp["size"] - cp["filter_1"] + 1
        cp["conv2_size"] = cp["conv1_size"] - cp["filter_2"] + 1
        netp["num_conv1"] = int(cp["conv1_size"]*cp["conv1_size"]*cp["channels_1"])
        netp["num_conv2"] = int(cp["conv2_size"]*cp["conv2_size"]*cp["channels_2"])

        #time_array = np.arange(0, tp["total_time"], tp["dt"])
        
        self.time_params, self.network_params, self.frame_params, self.convolution_params = tp, netp, fp, cp
        return netp["beta"], netp["num_rec"], netp["num_latent"], fp["depth"], netp["num_conv2"]

class CAE(nn.Module):
    def __init__(self, tp, netp, fp, cp, device, recurrence):
        super().__init__()
        self.device = device
        self.recurrence = recurrence
        
        # Process parameter dicts
        beta, num_rec, num_latent, depth, num_conv2 = self.process_params(tp, netp, fp, cp)
        
        # Convolution (encoder) - Input size: [batch_size, depth, 28, 28]
        self.conv1 = nn.Conv2d(depth, cp["channels_1"], (cp["filter_1"], cp["filter_1"]))
        self.conv2 = nn.Conv2d(cp["channels_1"], cp["channels_2"], (cp["filter_2"], cp["filter_2"]))
        
        # Recurrent (latent)
        self.ff_in = nn.Linear(num_conv2, num_rec)  # 64x24x24 (12288) -> 100
        self.ff_rec = nn.Linear(num_rec, num_rec)   # 100 -> 100 (same nodes)
        
        self.ff_out = nn.Linear(num_rec, num_conv2)
        
        # Convolution (decoder)
        self.deconv2 = nn.ConvTranspose2d(cp["channels_2"], cp["channels_1"], (cp["filter_2"], cp["filter_2"]))
        self.reconstruction = nn.ConvTranspose2d(cp["channels_1"], depth, (cp["filter_1"], cp["filter_1"]))
        
    def forward(self, x):
        num_rec, num_conv2, channels_1, conv1_size, channels_2, conv2_size = (
            self.network_params["num_rec"], self.network_params["num_conv2"],
            self.convolution_params["channels_1"], self.convolution_params["conv1_size"],
            self.convolution_params["channels_2"], self.convolution_params["conv2_size"]
        )
        print(x.shape)
        # Convolution (encoder) - layer 1
        x = F.relu(self.conv1(x))
        print(x.shape)
        # Convolution (encoder) - layer 2
        x = F.relu(self.conv2(x))
        print(x.shape)
        # Recurrent layer (encoder)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = F.relu(self.ff_in(x) + self.recurrence * self.ff_rec(x))
        print(x.shape)
        # Fully connected output
        x = F.relu(self.ff_out(x))
        print(x.shape)
        x = x.view(x.size(0), channels_2, conv2_size, conv2_size)
        print(x.shape)
        # Convolution (decoder) - undo layer 2
        x = F.relu(self.deconv2(x))
        print(x.shape)
        # Convolution (decoder) - undo layer 1
        x = F.relu(self.reconstruction(x))
        print(x.shape)
        return x
    
    def process_params(self, tp, netp, fp, cp):
        netp["beta"] = np.exp(-tp["dt"]/netp["tau_m"])
        netp["noise_amplitude"] = netp["eta"] * np.sqrt((1 - np.exp(-2*tp["dt"]/netp["tau_m"]))/2)
        tp["num_timesteps"] = int(tp["total_time"]/tp["dt"])
        
        cp["conv1_size"] = fp["size"] - cp["filter_1"] + 1
        cp["conv2_size"] = cp["conv1_size"] - cp["filter_2"] + 1
        netp["num_conv1"] = int(cp["conv1_size"] * cp["conv1_size"] * cp["channels_1"])
        netp["num_conv2"] = int(cp["conv2_size"] * cp["conv2_size"] * cp["channels_2"])
        
        self.time_params, self.network_params, self.frame_params, self.convolution_params = tp, netp, fp, cp
        return netp["beta"], netp["num_rec"], netp["num_latent"], fp["depth"], netp["num_conv2"]