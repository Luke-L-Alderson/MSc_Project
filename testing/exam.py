from .aux.dataset import H5Dataset
from .aux.functions import get_poisson_inputs, process_labels, latent_to_matrix, out_to_matrix
from torch.utils.data import DataLoader
import torch
import numpy as np


def test_network(spk_outs, labels, correct, total):
    width = 28
    M_min = 1
    M_max = 1
    max_count = torch.max(labels).cpu().detach().numpy()
    for data_point in range(spk_outs.shape[1]):
        true_matrix = latent_to_matrix(labels[data_point].cpu().detach().numpy()/max_count, M_max, width)[0]
        predicted_matrix = out_to_matrix(spk_outs[:, data_point, :].cpu().detach().numpy(), M_max, width)[0]
        correct["x"] -= np.abs(true_matrix["x"] - predicted_matrix["x"])
        correct["y"] -= np.abs(true_matrix["y"] - predicted_matrix["y"])
        if np.argmax(true_matrix["color"]) == np.argmax(predicted_matrix["color"]):
            correct[tuple(true_matrix["color"])] += 1
        total[tuple(true_matrix["color"])] += 1
        if true_matrix["shape"] == predicted_matrix["shape"]:
            correct[true_matrix["shape"]] += 1
        total[true_matrix["shape"]] += 1
    return correct, total


def get_exam_per_constant(network, input_specs, label_specs, exam_specs, device):
    recorded_vars = exam_specs["recorded_vars"]
    if "spk_out" not in recorded_vars:
        recorded_vars.append("spk_latent") 

    exams_dict = {}
    av_recorded_dict = {}

    for constant in exam_specs["constant_list"]:
        av_recorded = {var: [] for var in recorded_vars}
            
        test_dataset = H5Dataset('{}/test_{}.hdf5'.format(exam_specs["path"], constant))
        test_loader = DataLoader(test_dataset, batch_size=exam_specs["batch_size"], shuffle=True, drop_last=True)
        
        num_samples = len(test_dataset)
        num_batches = len(test_loader)
        
        

        correct = {"x": 0., "y": 0., (0, 0, 1): 0, (0, 1, 0): 0, (1, 0, 0): 0, "triangle": 0, "square": 0, "circle": 0}
        total = {(0, 0, 1): 0, (0, 1, 0): 0, (1, 0, 0): 0, "triangle": 0, "square": 0, "circle": 0}

        network.zero_grad()
        
        for batch_i, (test_inputs, test_labels) in enumerate(test_loader, 1):
            with torch.no_grad():
                inputs = get_poisson_inputs(test_inputs, **input_specs).to(device)
                labels = process_labels(test_labels, **label_specs).to(device).type(torch.cuda.FloatTensor)
                recorded  = network(inputs, recorded_vars=recorded_vars)
                correct, total = test_network(recorded["spk_latent"], labels, correct, total)
            

            if batch_i == 1:
                for var in recorded_vars:
                    av_recorded[var] = recorded[var].mean(1)/num_batches
            else:
                for var in recorded_vars:
                    av_recorded[var] += recorded[var].mean(1)/num_batches 
                    

        for key in correct.keys():
            if key in total:
                correct[key] = correct[key]/total[key] if total[key] != 0 else 0
            else:
                correct[key] = correct[key]/num_samples
        
        exams_dict[constant] = correct
        av_recorded_dict[constant] = av_recorded
        
    return exams_dict, av_recorded_dict