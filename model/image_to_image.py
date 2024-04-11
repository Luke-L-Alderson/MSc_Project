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

from brian2 import *

class SAE(nn.Module):
    def __init__(self, tp, netp, op, fp, cp, device):
        super().__init__()
        self.device = device
        
        #save and proces param dicts
        beta, num_rec, num_latent, depth, num_conv2 = self.process_params(tp, netp, op, fp, cp)
        
        #define gradient
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        '''
        self.encoder = nn.Sequential(
            nn.Conv2d(depth, cp["channels_1"], (cp["filter_1"], cp["filter_1"])),
            snn.Leaky(beta=beta, spike_grad=spike_grad),
            nn.Conv2d(cp["channels_1"], cp["channels_2"], (cp["filter_2"], cp["filter_2"])),
            snn.Leaky(beta=beta, spike_grad=spike_grad),

            #recurrent pool
            nn.Linear(num_conv2, num_rec),
            nn.Linear(num_rec, num_rec),
            snn.Leaky(beta=beta, spike_grad=spike_grad, reset_mechanism="zero"),

            # latent - do we want a spiking latent layer?
            nn.Linear(num_rec, num_latent),
            snn.Leaky(beta=beta, spike_grad=spike_grad)
            )
        
        self.decoder = nn.Sequential(
            # convolution (decoder)
            nn.ConvTranspose2d(cp["channels_2"], cp["channels_1"], (cp["filter_2"], cp["filter_2"])),
            snn.Leaky(beta=beta, spike_grad=spike_grad),
            nn.ConvTranspose2d(cp["channels_1"], depth, (cp["filter_1"], cp["filter_1"])),
            snn.Leaky(beta=beta, spike_grad=spike_grad)
            )
           
        '''    
        
        #convolution (encoder) - Input size: [8, 3, 28, 28]
        
        self.conv1 = nn.Conv2d(depth, cp["channels_1"], (cp["filter_1"], cp["filter_1"]))
        self.lif_conv1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        # conv1 output size: [8, 12, 26, 26]
        
        self.conv2 = nn.Conv2d(cp["channels_1"], cp["channels_2"], (cp["filter_2"], cp["filter_2"]))
        self.lif_conv2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        # conv2 output size: [8, 64, 24, 24]  

        # I think I should flatten conv2        

        # recurrent (latent)
        self.ff_in = nn.Linear(num_conv2, num_rec) # 8x24x64 (12288) -> 100
        self.ff_rec = nn.Linear(num_rec, num_rec)  # 100 -> 100 (same nodes)
        self.net_rec = snn.Leaky(beta=beta, spike_grad=spike_grad, reset_mechanism="zero")
        
        self.ff_out = nn.Linear(num_rec, num_conv2)
        # reshape the input to convTranspose to be 
        

        '''
        # latent - remove this
        self.ff_latent = nn.Linear(num_rec, num_latent) # 100 -> 8
        self.lif_latent = snn.Leaky(beta=beta, spike_grad=spike_grad)
        # latent output size: [8, 8]  
        '''
        
        # needs to be [8, 64, 24, 24] format for convTrans
        
        # convolution (decoder)
        
        self.deconv2 = nn.ConvTranspose2d(cp["channels_2"], cp["channels_1"], (cp["filter_2"], cp["filter_2"])) 
        self.lif_deconv2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        # deconv2 output size: [8, 12, 26, 26]  
        
        self.reconstruction = nn.ConvTranspose2d(cp["channels_1"], depth, (cp["filter_1"], cp["filter_1"]))
        self.lif_reconstruction = snn.Leaky(beta=beta, spike_grad=spike_grad)
        # recontruction should be [8, 3, 28, 28]
        
        
    def forward(self, x, recorded_vars=None):
        
        #seed = np.random.randint(1e6)
        #torch.cuda.manual_seed_all(seed)

        #dict variables
        num_rec, num_conv2, noise_amp, osc_of_t, channels_1, conv1_size, channels_2, conv2_size = self.network_params["num_rec"], self.network_params["num_conv2"], self.network_params["noise_amplitude"], self.oscillation_params["osc_of_t"], self.convolution_params["channels_1"], self.convolution_params["conv1_size"], self.convolution_params["channels_2"], self.convolution_params["conv2_size"]

        try:
          x = x.type(torch.cuda.FloatTensor)
        except:
          x = x.type(torch.FloatTensor)
        
        # Initialize hidden states and outputs at t=0
        
        batch_size = len(x[0])       
        mem_conv1 = self.lif_conv1.init_leaky()
        mem_conv2 = self.lif_conv2.init_leaky()
        mem_rec = self.net_rec.init_leaky()
        #mem_latent = self.lif_latent.init_leaky()
        mem_deconv2 = self.lif_deconv2.init_leaky()
        mem_reconstruction = self.lif_reconstruction.init_leaky()
        spk_conv1 = torch.zeros(batch_size, channels_1, conv1_size, conv1_size).to(self.device)
        spk_conv2 = torch.zeros(batch_size, channels_2, conv2_size, conv2_size).to(self.device)
        spk_deconv1 = torch.zeros(batch_size, channels_2, conv2_size, conv2_size).to(self.device)
        spk_deconv2 = torch.zeros(batch_size, channels_1, conv1_size, conv1_size).to(self.device)
        spk_rec = torch.zeros(batch_size, num_rec).to(self.device)
        
        spk_outs = torch.zeros(batch_size, channels_2, conv2_size, conv2_size).to(self.device)
        
        spk_outs, spk_recs = [], []
        
        # Record latent and output layer
        if recorded_vars:
            recorded = {key: [] for key in recorded_vars}

        for timestep in range(self.time_params["num_timesteps"]):
            
            '''
            Questions for discussion:
                
                1. Are we interested in noise?
                2. Do we still need the recurrent layer?
                3. Should I remove oscillations?
            
            The below code passes the signal x[timestep] through a number of neuron layers, as follows:
                
                1. Convolution + LIF
                2. Convolution + LIF
                3. Reccurrent LIF (i.e. input + recurrent)
                4. Latent LIF
                5. Deonvolution + LIF
                6. Deonvolution + LIF
    
            '''
            
            curr_osc = osc_of_t[timestep]
            
            # convolution (encoder) - layer 1
            curr_conv1 = self.conv1(x[timestep])
            spk_conv1, mem_conv1 = self.lif_conv1(curr_conv1 + curr_osc, mem_conv1)
            
            # convolution (encoder) - layer 2
            curr_conv2 = self.conv2(spk_conv1)
            spk_conv2, mem_conv2 = self.lif_conv2(curr_conv2 + curr_osc, mem_conv2)
            
            
            #recurrent layer (encoder)
            curr_in = self.ff_in(spk_conv2.view(batch_size, -1))
            curr_rec = self.ff_rec(spk_rec)
            curr_total = curr_in + curr_rec
            spk_rec, mem_rec = self.net_rec(curr_total + curr_osc, mem_rec)
            '''
            print(f'\nInput Size: {x[timestep].size()}\
                    \nConv1 Size: {curr_conv1.size()}\
                    \nSpk1 Size: {spk_conv1.size()}\
                    \nConv2 Size: {curr_conv2.size()}\
                    \nSpk2 Size {spk_conv2.size()}\
                    \nSpkR Size {spk_rec.size()}')
            '''
            
            curr_out = self.ff_out(spk_rec)
            
            # convolution (decoder) - undo layer 2
            curr_deconv2 = self.deconv2(curr_out.view(curr_out.size(0), channels_2, conv2_size, conv2_size))
            spk_deconv2, mem_deconv2 = self.lif_conv2(curr_deconv2 + curr_osc, mem_deconv2)
            #print(f'\n{spk_deconv2.size()}\n{curr_deconv2.size()}')
            
            # convolution (decoder) - undo layer 1
            curr_reconstruction = self.reconstruction(spk_deconv2)
            spk_reconstruction, mem_reconstruction = self.lif_reconstruction(curr_reconstruction + curr_osc, mem_reconstruction)
            #print(f'\n{spk_reconstruction.size()}\n{curr_reconstruction.size()}')
            
            ''' Noise if needed
            #mem_conv1 += noise_amp*torch.randn(mem_conv1.shape).to(self.device)
            #mem_conv2 += noise_amp*torch.randn(mem_conv2.shape).to(self.device)
            #mem_rec += noise_amp*torch.randn(mem_rec.shape).to(self.device)
            #mem_latent += noise_amp*torch.randn(mem_latent.shape).to(self.device)
            #mem_deconv2 += noise_amp*torch.randn(mem_deconv2.shape).to(self.device)
            #mem_reconstruction += noise_amp*torch.randn(mem_reconstruction.shape).to(self.device)
            '''
            
            #spk_latents.append(spk_latent)
            spk_recs.append(spk_rec)
            spk_outs.append(spk_reconstruction)

            if recorded_vars:
                for key in recorded:
                    recorded[key].append(locals()[key])
        
        if recorded_vars:
            for key, item in recorded.items():
                recorded[key] = torch.stack(item)
            return recorded
        return torch.stack(spk_recs), torch.stack(spk_outs)
    
    
    def process_params(self, tp, netp, op, fp, cp):
        netp["beta"] = np.exp(-tp["dt"]/netp["tau_m"])
        netp["noise_amplitude"] = netp["eta"]*np.sqrt((1 - np.exp(-2*tp["dt"]/netp["tau_m"]))/2)
        tp["num_timesteps"] = int(tp["total_time"]/tp["dt"])
        
        cp["conv1_size"] = fp["size"] - cp["filter_1"] + 1
        cp["conv2_size"] = cp["conv1_size"] - cp["filter_2"] + 1
        netp["num_conv1"] = int(cp["conv1_size"]*cp["conv1_size"]*cp["channels_1"])
        netp["num_conv2"] = int(cp["conv2_size"]*cp["conv2_size"]*cp["channels_2"])
        
        op["T"] = 1/op["f"]
        op["omega"] = 2*np.pi*op["f"]
        time_array = np.arange(0, tp["total_time"], tp["dt"])
        op["osc_of_t"] = torch.tensor(netp["R_m"]*op["I_osc"]/netp["v_th"]*(1 - netp["beta"])*(-1 + np.cos(op["omega"]*time_array - np.pi))/2)
        
        self.time_params, self.network_params, self.oscillation_params, self.frame_params, self.convolution_params = tp, netp, op, fp, cp
        return netp["beta"], netp["num_rec"], netp["num_latent"], fp["depth"], netp["num_conv2"]