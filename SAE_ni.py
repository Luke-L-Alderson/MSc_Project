import torch
import torch.nn as nn
import numpy as np
import snntorch as snn
from snntorch import surrogate

class SAE_ni(nn.Module):
    def __init__(self, tp, netp, fp, cp, device, recurrence):
        super().__init__()
        self.device = device
        self.recurrence = recurrence
        beta, num_rec, depth, num_conv2 = self.process_params(tp, netp, fp, cp)
        threshold = netp["v_th"]
        learnable = netp["learnable"]
        
        spike_grad = surrogate.fast_sigmoid(slope=25)
          
        self.conv1 = nn.Conv2d(depth, cp["channels_1"], (cp["filter_1"], cp["filter_1"]))
        self.lif_conv1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold, init_hidden=True)
        self.lif_rec = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold, init_hidden=True)
        self.conv2 = nn.Conv2d(cp["channels_1"], cp["channels_2"], (cp["filter_2"], cp["filter_2"]))
        self.lif_conv2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold, init_hidden=True)
        self.ff_in = nn.Linear(num_conv2, num_rec)
        self.lif_ff_out = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold, init_hidden=True)
        self.ff_out = nn.Linear(num_rec, num_conv2)     
        self.deconv2 = nn.ConvTranspose2d(cp["channels_2"], cp["channels_1"], (cp["filter_2"], cp["filter_2"])) 
        self.lif_deconv2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold, init_hidden=True)
        self.reconstruction = nn.ConvTranspose2d(cp["channels_1"], depth, (cp["filter_1"], cp["filter_1"]))
        self.lif_reconstruction = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold, init_hidden=True)
        self.rlif_rec = snn.RLeaky(beta=beta, 
                                   spike_grad=spike_grad, 
                                   threshold = threshold,
                                   all_to_all = True, 
                                   learn_recurrent=learnable,  
                                   linear_features=num_rec,
                                   init_hidden=True
                                   )
        
    def forward(self, x, recorded_vars=None):
        channels_2, conv2_size = self.convolution_params["channels_2"], self.convolution_params["conv2_size"]

        try:
          x = torch.cuda.FloatTensor(x)
        except:
          x = torch.FloatTensor(x)
        
        batch_size = x.shape[0]

        curr_conv1 = self.conv1(x)
        spk_conv1 = self.lif_conv1(curr_conv1)
        
        curr_conv2 = self.conv2(spk_conv1)
        spk_conv2 = self.lif_conv2(curr_conv2)
        
        curr_in = self.ff_in(spk_conv2.view(batch_size, -1))
        
        spk_rec = self.rlif_rec(curr_in) if self.recurrence else self.lif_rec(curr_in)
            
        curr_out = self.ff_out(spk_rec)
        spk_out = self.lif_ff_out(curr_out)
        
        curr_deconv2 = self.deconv2(spk_out.view(-1, channels_2, conv2_size, conv2_size))
        spk_deconv2 = self.lif_deconv2(curr_deconv2)
        
        curr_reconstruction = self.reconstruction(spk_deconv2)
        spk_reconstruction = self.lif_reconstruction(curr_reconstruction)
        
        return spk_rec, spk_reconstruction
    
    
    def process_params(self, tp, netp, fp, cp):
        netp["beta"] = np.exp(-tp["dt"]/netp["tau_m"])
        netp["noise_amplitude"] = netp["eta"]*np.sqrt((1 - np.exp(-2*tp["dt"]/netp["tau_m"]))/2)
        tp["num_timesteps"] = int(tp["total_time"]/tp["dt"])
        
        cp["conv1_size"] = fp["size"] - cp["filter_1"] + 1
        cp["conv2_size"] = cp["conv1_size"] - cp["filter_2"] + 1
        netp["num_conv1"] = int(cp["conv1_size"]*cp["conv1_size"]*cp["channels_1"])
        netp["num_conv2"] = int(cp["conv2_size"]*cp["conv2_size"]*cp["channels_2"])
        
        self.time_params, self.network_params, self.frame_params, self.convolution_params = tp, netp, fp, cp
        return netp["beta"], netp["num_rec"], fp["depth"], netp["num_conv2"]

