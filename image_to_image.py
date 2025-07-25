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
        beta, num_rec, depth, num_conv2 = self.process_params(tp, netp, fp, cp)
        threshold = netp["v_th"]
        learnable = True
        
        spike_grad = surrogate.fast_sigmoid()
          
        self.conv1 = nn.Conv2d(depth, cp["channels_1"], (cp["filter_1"], cp["filter_1"]))
        self.lif_conv1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold)
        
        self.conv2 = nn.Conv2d(cp["channels_1"], cp["channels_2"], (cp["filter_2"], cp["filter_2"]))
        self.lif_conv2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold)
        
        self.ff_in = nn.Linear(num_conv2, num_rec)
        
        if self.recurrence:
            self.rlif_rec = snn.RLeaky(beta=beta, 
                                       spike_grad=spike_grad, 
                                       threshold = threshold,
                                       all_to_all = True, 
                                       learn_recurrent=learnable,  
                                       linear_features=num_rec,
                                       )
        else:
            self.lif_rec = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold)
            self.ff_in2 = nn.Linear(num_rec, num_rec)
            self.lif_rec2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold)
        
        self.ff_out = nn.Linear(num_rec, num_conv2)   
        self.lif_ff_out = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold)
        
        self.deconv2 = nn.ConvTranspose2d(cp["channels_2"], cp["channels_1"], (cp["filter_2"], cp["filter_2"])) 
        self.lif_deconv2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold)
        
        self.reconstruction = nn.ConvTranspose2d(cp["channels_1"], depth, (cp["filter_1"], cp["filter_1"]))
        self.lif_reconstruction = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold)

        
    def forward(self, x, recorded_vars=None):
        channels_2, conv2_size = self.convolution_params["channels_2"], self.convolution_params["conv2_size"]

        try:
          x = x.type(torch.cuda.FloatTensor)
        except:
          x = x.type(torch.FloatTensor)

        batch_size = x.shape[1]
        time_steps = x.shape[0]

        mem_conv1 = self.lif_conv1.init_leaky()
        mem_out = self.lif_ff_out.init_leaky()
        mem_conv2 = self.lif_conv2.init_leaky()
        
        if self.recurrence:
            spk_rec, mem_rec = self.rlif_rec.init_rleaky()
        else:
            mem_rec1 = self.lif_rec.init_leaky()
            mem_rec = self.lif_rec2.init_leaky()
            
        mem_deconv2 = self.lif_deconv2.init_leaky()
        mem_reconstruction = self.lif_reconstruction.init_leaky()
        spk_outs, spk_recs = [], []

        for timestep in range(time_steps):
            curr_conv1 = self.conv1(x[timestep])
            spk_conv1, mem_conv1 = self.lif_conv1(curr_conv1, mem_conv1)
            
            curr_conv2 = self.conv2(spk_conv1)
            spk_conv2, mem_conv2 = self.lif_conv2(curr_conv2, mem_conv2)
            
            curr_in = self.ff_in(spk_conv2.view(batch_size, -1))
            
            if self.recurrence:
                spk_rec, mem_rec = self.rlif_rec(curr_in, spk_rec, mem_rec)
            else:
                spk_rec, mem_rec1 = self.lif_rec(curr_in, mem_rec1)
            
            curr_out = self.ff_out(spk_rec)
            
            spk_out, mem_out = self.lif_ff_out(curr_out, mem_out)
            
            curr_deconv2 = self.deconv2(spk_out.view(spk_out.size(0), channels_2, conv2_size, conv2_size))
            spk_deconv2, mem_deconv2 = self.lif_deconv2(curr_deconv2, mem_deconv2)
            
            curr_reconstruction = self.reconstruction(spk_deconv2)
            spk_reconstruction, mem_reconstruction = self.lif_reconstruction(curr_reconstruction, mem_reconstruction)
    

            spk_recs.append(spk_rec)
            spk_outs.append(spk_reconstruction)
        
        return torch.stack(spk_recs), torch.stack(spk_outs)
    
    
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

class CAE(nn.Module):
    def __init__(self, num_rec, cp, recurrence):
        super().__init__()
        self.recurrence = recurrence
        self.convolution_params = cp
        im_dim = 28
        cp["conv1_size"] = im_dim - cp["filter_1"] + 1
        cp["conv2_size"] = cp["conv1_size"] - cp["filter_2"] + 1
        num_conv2 = int(cp["conv2_size"]*cp["conv2_size"]*cp["channels_2"])
        depth = 1
        
        # Convolution (encoder) - Input size: [batch_size, depth, 28, 28]
        self.conv1 = nn.Conv2d(depth, cp["channels_1"], (cp["filter_1"], cp["filter_1"]))
        self.conv2 = nn.Conv2d(cp["channels_1"], cp["channels_2"], (cp["filter_2"], cp["filter_2"]))
        self.ff_in = nn.Linear(num_conv2, num_rec)  # 64x24x24 (12288) -> 100
        self.ff_rec = nn.Linear(num_rec, num_rec)   # 100 -> 100 (same nodes)
        self.ff_out = nn.Linear(num_rec, num_conv2)
        self.deconv2 = nn.ConvTranspose2d(cp["channels_2"], cp["channels_1"], (cp["filter_2"], cp["filter_2"]))
        self.reconstruction = nn.ConvTranspose2d(cp["channels_1"], depth, (cp["filter_1"], cp["filter_1"]))
        
    def forward(self, x):
        channels_2, conv2_size = self.convolution_params["channels_2"], self.convolution_params["conv2_size"]
        try:
          x = x.type(torch.cuda.FloatTensor)
        except:
          x = x.type(torch.FloatTensor)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.ff_in(x))
        x_out = F.relu(self.ff_out(x))
        x_out = x_out.view(x_out.size(0), channels_2, conv2_size, conv2_size)
        x_out = F.relu(self.deconv2(x_out))
        x_out = F.relu(self.reconstruction(x_out))

        return x, x_out
    
class SAE_ni(nn.Module):
    def __init__(self, tp, netp, fp, cp, device, recurrence):
        super().__init__()
        self.device = device
        self.recurrence = recurrence
        beta, num_rec, depth, num_conv2 = self.process_params(tp, netp, fp, cp)
        threshold = netp["v_th"]
        
        spike_grad = surrogate.fast_sigmoid(slope=25)
          
        self.conv1 = nn.Conv2d(depth, cp["channels_1"], (cp["filter_1"], cp["filter_1"]))
        self.lif_conv1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold, init_hidden=True)

        self.conv2 = nn.Conv2d(cp["channels_1"], cp["channels_2"], (cp["filter_2"], cp["filter_2"]))
        self.lif_conv2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold, init_hidden=True)
        
        self.ff_in = nn.Linear(num_conv2, num_rec)
        if not self.recurrence:
            self.lif_rec = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold, init_hidden=True)
            self.ff_in2 = nn.Linear(num_rec, num_rec)
            self.lif_rec2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold, init_hidden=True)
        else:
            self.rlif_rec = snn.RLeaky(beta=beta, 
                                        spike_grad=spike_grad, 
                                        threshold = threshold,
                                        all_to_all = True,
                                        linear_features=num_rec,
                                        init_hidden=True
                                        )

        
        self.ff_out = nn.Linear(num_rec, num_conv2)  
        self.lif_ff_out = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold, init_hidden=True)
           
        self.deconv2 = nn.ConvTranspose2d(cp["channels_2"], cp["channels_1"], (cp["filter_2"], cp["filter_2"])) 
        self.lif_deconv2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold, init_hidden=True)
        
        self.reconstruction = nn.ConvTranspose2d(cp["channels_1"], depth, (cp["filter_1"], cp["filter_1"]))
        self.lif_reconstruction = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold = threshold, init_hidden=True)

        
        
    def forward(self, x, recorded_vars=None):
        channels_2, conv2_size = self.convolution_params["channels_2"], self.convolution_params["conv2_size"]
        
        try:
          x = x.type(torch.cuda.FloatTensor)
        except:
          x = x.type(torch.FloatTensor)
        
        batch_size = x.shape[1]
        time_steps = x.shape[0]
        spk_recs, spk_outs = [], []
        
        for timestep in range(time_steps):
            curr_conv1 = self.conv1(x[timestep])
            spk_conv1 = self.lif_conv1(curr_conv1)
            
            curr_conv2 = self.conv2(spk_conv1)
            spk_conv2 = self.lif_conv2(curr_conv2)
            
            curr_in = self.ff_in(spk_conv2.view(batch_size, -1))
            
            if self.recurrence:
                spk_rec = self.rlif_rec(curr_in)
            else:
                spk_rec = self.lif_rec(curr_in)
            
            curr_out = self.ff_out(spk_rec)
            
            spk_out = self.lif_ff_out(curr_out)
            
            curr_deconv2 = self.deconv2(spk_out.view(-1, channels_2, conv2_size, conv2_size))
            spk_deconv2 = self.lif_deconv2(curr_deconv2)
            
            curr_reconstruction = self.reconstruction(spk_deconv2)
            spk_reconstruction = self.lif_reconstruction(curr_reconstruction)
            
            spk_recs.append(spk_rec)
            spk_outs.append(spk_reconstruction)
    
        return torch.stack(spk_recs), torch.stack(spk_outs)
    
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
    
class SC(nn.Module):
    def __init__(self, tp, netp, fp, cp, device, recurrence):
        super().__init__()
        self.device = device
        self.recurrence = recurrence
        num_classes = 11
        beta, num_rec, depth, num_conv2 = self.process_params(tp, netp, fp, cp)
        threshold = netp["v_th"]

        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.conv1 = nn.Conv2d(depth, cp["channels_1"], (cp["filter_1"], cp["filter_1"]))
        self.lif_conv1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold, init_hidden=True)

        self.conv2 = nn.Conv2d(cp["channels_1"], cp["channels_2"], (cp["filter_2"], cp["filter_2"]))
        self.lif_conv2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold, init_hidden=True)

        self.ff_in = nn.Linear(num_conv2, num_rec)
        if not self.recurrence:
            self.lif_rec = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold, init_hidden=True)
            self.ff_in2 = nn.Linear(num_rec, num_rec)
            self.lif_rec2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold, init_hidden=True)
        else:
            self.rlif_rec = snn.RLeaky(beta=beta, 
                                        spike_grad=spike_grad, 
                                        threshold=threshold,
                                        all_to_all=True,
                                        linear_features=num_rec,
                                        init_hidden=True)

        self.ff_out = nn.Linear(num_rec, num_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold, init_hidden=True, output=True)

    def forward(self, x, recorded_vars=None):
        
        try:
            x = x.type(torch.cuda.FloatTensor)
        except:
            x = x.type(torch.FloatTensor)
        
        batch_size = x.shape[1]
        time_steps = x.shape[0]
        
        spk_outs, mem_outs, = [], []
        
        for timestep in range(time_steps):
            curr_conv1 = self.conv1(x[timestep])
            spk_conv1 = self.lif_conv1(curr_conv1)
            
            curr_conv2 = self.conv2(spk_conv1)
            spk_conv2 = self.lif_conv2(curr_conv2)
            
            curr_in = self.ff_in(spk_conv2.view(batch_size, -1))
            
            if self.recurrence:
                spk_rec = self.rlif_rec(curr_in)
            else:
                spk_rec1 = self.lif_rec(curr_in)
                curr_in2 = self.ff_in2(spk_rec1)
                spk_rec = self.lif_rec2(curr_in2)
            
            # Final output directly from the last linear layer
            curr_out = self.ff_out(spk_rec)  # Obtain logits directly from the final linear layer
            spk_out, mem_out = self.lif_out(curr_out)
            
            mem_outs.append(mem_out)
            spk_outs.append(spk_out)
        
        return torch.stack(spk_outs), torch.stack(mem_outs)

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