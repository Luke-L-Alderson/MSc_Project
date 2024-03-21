import snntorch as snn
import torch
import torch.nn as nn

def get_poisson_inputs(inputs, total_time, bin_size, rate_on, rate_off):
    num_steps = int(total_time/bin_size)
    bin_prob_on = rate_on*bin_size
    bin_prob_off = rate_off*bin_size
    poisson_input = snn.spikegen.rate((bin_prob_on - bin_prob_off)*inputs + bin_prob_off*torch.ones(inputs.shape), num_steps=num_steps)
    return poisson_input


def process_labels(labels, total_time, code, rate=None):
    if code == 'rate':
        labels = labels*(rate*total_time)
    return labels


class mse_count_loss():
    def __init__(
        self, lambda_rate, lambda_weights
    ):  
        self.lambda_r = lambda_rate
        self.lambda_w = lambda_weights
        self.__name__ = "mse_count_loss"
        
    def __call__(self, spk_recs, spk_outs, targets):
        spike_count = torch.sum(spk_outs, 0)
        loss_fn = nn.MSELoss()
        max_count = torch.max(targets)
        loss = loss_fn(spike_count, targets) + self.lambda_r*torch.sum(spk_recs)
        return loss/max_count