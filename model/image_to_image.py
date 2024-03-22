import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop

from brian2 import *


class autoNet(nn.Module):
    def __init__(self, tp, netp, op, fp, cp, device):
        super().__init__()
        self.device = device
        
        #save and proces param dicts
        beta, num_rec, num_latent, depth, num_conv2 = self.process_params(tp, netp, op, fp, cp)
        
        #define gradient
        spike_grad = surrogate.fast_sigmoid(slope=25)

        #convolution (encoder)
        self.conv1 = nn.Conv2d(depth, cp["channels_1"], (cp["filter_1"], cp["filter_1"]))
        #self.ff_rec_conv1 = nn.Linear(num_conv1, num_conv1)
        self.lif_conv1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(cp["channels_1"], cp["channels_2"], (cp["filter_2"],cp["filter_2"]))
        #self.ff_rec_conv2 = nn.Linear(num_conv2, num_conv2)
        self.lif_conv2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        #recurrent pool
        self.ff_in = nn.Linear(num_conv2, num_rec)
        self.ff_rec = nn.Linear(num_rec, num_rec)
        self.net_rec = snn.Leaky(beta=beta, spike_grad=spike_grad, reset_mechanism="zero")

        #convolution (decoder)
        self.ff_latent = nn.Linear(num_rec, num_latent)
        self.lif_latent = snn.Leaky(beta=beta, spike_grad=spike_grad)

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
        mem_conv1 = self.lif_conv1.init_leaky()
        mem_conv2 = self.lif_conv2.init_leaky()
        mem_rec = self.net_rec.init_leaky()
        mem_latent = self.lif_latent.init_leaky()
        batch_size = len(x[0])
        spk_conv1 = torch.zeros(batch_size, channels_1, conv1_size, conv1_size).to(self.device)
        spk_conv2 = torch.zeros(batch_size, channels_2, conv2_size, conv2_size).to(self.device)
        spk_rec = torch.zeros(batch_size, num_rec).to(self.device)
        
        batch_size = len(x[0])
        
        spk_recs, spk_outs = [], []
        
        # Record recurrent and output layer
        if recorded_vars:
            recorded = {key: [] for key in recorded_vars}

        for timestep in range(self.time_params["num_timesteps"]):
            
            curr_osc = osc_of_t[timestep]
            
            #convolution
            #curr_conv1 = self.conv1(x) + self.ff_rec_conv1(spk_conv1.view(batch_size, -1)).view(batch_size, channels_1, conv1_size, conv1_size)
            curr_conv1 = self.conv1(x[timestep])
            spk_conv1, mem_conv1 = self.lif_conv1(curr_conv1 + curr_osc, mem_conv1)
            mem_conv1 += noise_amp*torch.randn(mem_conv1.shape).to(self.device)
            
            #curr_conv2 = self.conv2(spk_conv1) + self.ff_rec_conv2(spk_conv2.view(batch_size, -1)).view(batch_size, channels_2, conv2_size, conv2_size)
            curr_conv2 = self.conv2(spk_conv1)
            spk_conv2, mem_conv2 = self.lif_conv2(curr_conv2 + curr_osc, mem_conv2)
            mem_conv2 += noise_amp*torch.randn(mem_conv2.shape).to(self.device)

            #recurrent
            curr_in = self.ff_in(spk_conv2.view(batch_size, -1))
            curr_rec = self.ff_rec(spk_rec)
            #print(torch.mean(torch.abs(curr_in)),torch.mean(torch.abs(noise_amp*torch.randn(curr_in.shape).to(device))))
            curr_total = curr_in + curr_rec
            spk_rec, mem_rec = self.net_rec(curr_total + curr_osc, mem_rec)
            mem_rec += noise_amp*torch.randn(mem_rec.shape).to(self.device)

            #convolution (decoder)
            curr_latent = self.ff_latent(spk_rec)
            spk_latent, mem_latent = self.lif_latent(curr_latent + curr_osc, mem_latent)
            mem_latent += noise_amp*torch.randn(mem_latent.shape).to(self.device)


            #append recordings in time axis
            spk_recs.append(spk_rec)
            spk_outs.append(spk_latent)

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