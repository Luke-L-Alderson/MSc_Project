"""github_testing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IMVe2ipvPRsU3l9Codli-01onzhT_3YU

# Training a Spiking CNN via Backrpopagation

## Generate GitHub keys and clone repository
"""## Imports"""

#importing module
import sys
sys.path.append('snn-project')
import os, shutil
import torch

from torch.utils.data import DataLoader
import snntorch as snn
import snntorch.spikeplot as splt
from IPython.display import HTML
from brian2 import *
import seaborn as sns

print(torch.cuda.is_available())

from model.train.train_network import train_network
from model.image_to_latent import Net
from model.aux.functions import get_poisson_inputs, process_labels, mse_count_loss
from data.aux.dataset import H5Dataset
from data_generator.ds_generator import make_dataset, make_exam_tests
from testing.exam import get_exam_per_constant


dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def to_np(tensor):
  return tensor.detach().cpu().numpy()

def plot_input(inputs, index):
  plt.imshow(to_np(torch.transpose(torch.sum(inputs, 0)[index], 0 ,2)))

def curr_to_pA(curr, network):
  factor = network.network_params["v_th"]/network.network_params["R_m"]/(1 - network.network_params["beta"])
  try:
    return to_np(curr)*factor
  except:
    return curr*factor

def transfer(curr, network):
  T = -network.network_params["tau_m"]*np.log(1 - network.network_params["v_th"]/(curr*network.network_params["R_m"]))
  return np.clip(1/T, 0*Hz, inf*Hz)

def get_fr(raster, network):
  return to_np(torch.sum(raster, 0))/network.time_params["total_time"]

#%%
"""## Make or access existing datasets"""
folder = 'data/content'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
        
# create dataset in /content
make_dataset(train_samples=3000, test_samples=300,  width=28, M_min=1, M_max=1, restrictions={}, path=folder)
make_exam_tests(test_samples=1000, width=28, M_min=1, M_max=1, path=folder)

#load dataset
train_dataset = H5Dataset('data/world_data_1_1/train.hdf5')
test_dataset = H5Dataset('data/world_data_1_1/test.hdf5')

# Create DataLoader
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

"""## Define network architecutre and parameters"""

time_params = {}
time_params["dt"] = 1*ms
time_params["total_time"] = 200*ms

network_params = {}
network_params["tau_m"] = 24*ms
network_params["tau_syn"] = 10*ms
network_params["R_m"] = 146*Mohm
network_params["v_th"] = 15*mV
network_params["eta"] = 0.0
network_params["num_rec"] = 100
network_params["num_latent"] = 8

oscillation_params = {}
oscillation_params["f"] = 10*Hz
oscillation_params["I_osc"] = 0*pA
#oscillation_params["phis"] = [0, 0, 0, 0]

frame_params = {}
frame_params["depth"] = 3
frame_params["size"] = 28

convolution_params = {}
convolution_params["channels_1"] = 12
convolution_params["filter_1"] = 3
convolution_params["channels_2"] = 64
convolution_params["filter_2"] = 3

network = Net(time_params, network_params, oscillation_params, frame_params, convolution_params, device).to(device)
print_network_architecure(network)

def print_network_architecure(network):
    netp, op, fp, cp = network.network_params, network.oscillation_params, network.frame_params, network.convolution_params
    input_layer_text = """
    Input layer: {} channels
                {}x{} neurons/channel
                {} total neurons
    """.format(fp["depth"], fp["size"], fp["size"],fp["depth"]*fp["size"]*fp["size"] )

    conv1_text = """
    Conv1 layer: {} channels
                {}x{} neurons/channel
                {} total neurons
                {} synapses/neuron (ff)
                {} total_synapses
    """.format(cp["channels_1"], cp["conv1_size"], cp["conv1_size"], netp["num_conv1"], cp["filter_1"]*cp["filter_1"], netp["num_conv1"]*cp["filter_1"]*cp["filter_1"])

    conv2_text = """
    Conv2 layer: {} channels
                {}x{} neurons/channel
                {} total neurons
                {} synapses/neuron (ff)
                {} total_synapses
    """.format(cp["channels_2"], cp["conv2_size"], cp["conv2_size"], netp["num_conv2"], cp["filter_2"]*cp["filter_2"], netp["num_conv2"]*cp["filter_2"]*cp["filter_2"])

    rec_text = """
    Rec layer:   {} total neurons
                {} synapses/neuron (ff) and {} synapses/neuron (rec)
                {} total_synapses
    """.format(netp["num_rec"], netp["num_conv2"], netp["num_rec"], netp["num_conv2"]*netp["num_rec"] + netp["num_rec"]**2)

    latent_text = """
    Lat layer:   {} total neurons
                {} synapses/neuron (ff)
                {} total_synapses
    """.format(netp["num_latent"], netp["num_rec"], netp["num_rec"], netp["num_rec"]*netp["num_latent"])

    Trec_text = ""
    Tconv2_text = ""
    Tconv1_text = ""
    output_layer_text = ""

    print(input_layer_text)
    print(conv1_text)
    print(conv2_text)
    print(rec_text)
    print(latent_text)
    print(Trec_text)
    print(Tconv2_text)
    print(Tconv1_text)
    print(output_layer_text)

#%%
"""## Training the network"""

# set up hyperparams

input_specs = {}
label_specs = {}
train_specs = {}

input_specs["total_time"] = 200*ms
input_specs["bin_size"] = 1*ms
input_specs["rate_on"] = 75*Hz
input_specs["rate_off"] = 10*Hz

label_specs["total_time"] = 200*ms
label_specs["code"] = 'rate'
label_specs["rate"] = 75*Hz

train_specs["num_epochs"] = 1
train_specs["early_stop"] = -1
train_specs["device"] = device
train_specs["lr"] = 1e-3
train_specs["loss_fn"] = "spike_count"
train_specs["lambda_rate"] = 0.0
train_specs["lambda_weights"] = None

exam_specs = {}
exam_specs["constant_list"] = ["none"]
exam_specs["batch_size"] = 16
exam_specs["recorded_vars"] = ["curr_conv1", "spk_conv1", "curr_conv2", "spk_conv2", "curr_total", "spk_rec", "curr_latent", "spk_latent"]
exam_specs["path"] = 'data/content/exam_world_data_1_1'

num_epochs = 1
for epoch in range(num_epochs):
  network = train_network(network, train_loader, test_loader, input_specs, label_specs, train_specs)
  exams_dict, av_recorded_dict = get_exam_per_constant(network, input_specs, label_specs, exam_specs, device)
  print(f'Epoch: {epoch} - {exams_dict["none"]}')


# what are these lines? - I think the first line saves a network, the second
# loads the settings so you dont have to keep training

#torch.save(network.state_dict(), 'data/content/pt_model_10_500.pth')
#network.load_state_dict(torch.load('pt_model_10_500.pth'), strict=False)

#%%
"""## Explore physiological statistics"""

input_specs["rate_on"] = 500*Hz
input_specs["rate_off"] = 10*Hz

inputs, labels = next(iter(train_loader))
poisson_inputs = get_poisson_inputs(inputs, **input_specs).to(device)

input_index = 0
plt.imshow(to_np(torch.transpose(inputs[input_index], 0 ,2)))

fig, ax = plt.subplots()
anim = splt.animator(torch.transpose(torch.sum(poisson_inputs[:, input_index], 1), 1, 2), fig, ax)
HTML(anim.to_html5_video())

input_specs["rate_on"] = 0*Hz
input_specs["rate_off"] = 0*Hz
recorded_vars=["mem_conv1", "mem_conv2", "mem_rec", "mem_latent"]

traj_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
inputs, labels = next(iter(traj_loader))
traj_inputs = get_poisson_inputs(inputs, **input_specs).to(device)

num_trials = 100
trajectories = {var : [] for var in recorded_vars}
index = {}

index["mem_conv1"] = 26*26*2 + 1
plt.plot(to_np(recordings["mem_conv1"].view(int(network.time_params["total_time"]/ms), -1)[:, index["mem_conv1"]]))

index["mem_conv2"] = 24*24*8 + 1
plt.plot(to_np(recordings["mem_conv2"].view(int(network.time_params["total_time"]/ms), -1)[:, index["mem_conv2"]]))

index["mem_rec"] = 7
plt.plot(to_np(recordings["mem_rec"].view(int(network.time_params["total_time"]/ms), -1)[:, index["mem_rec"]]))

index["mem_latent"] = 5
plt.plot(to_np(recordings["mem_latent"].view(int(network.time_params["total_time"]/ms), -1)[:, index["mem_latent"]]))

for trial in range(num_trials):
  print(f'Getting Recordings: {trial+1} / {num_trials}')
  recordings = network(traj_inputs, recorded_vars=recorded_vars)
  for var in recorded_vars:
    trajectories[var].append(to_np(recordings[var].view(int(network.time_params["total_time"]/ms), -1)[:, index[var]])*15)

t_values = np.arange(0, network.time_params["total_time"], step=network.time_params["dt"])
var_th = (1 - np.exp(-t_values/network.network_params["tau_m"]))*network.network_params["eta"]*network_params["v_th"]/mV
fig, axs = plt.subplots(1, len(recorded_vars), sharey=True)
fig.suptitle(r"$u$ trajectories (addition of noise)", y= 1.05, fontsize=20)
fig.set_size_inches(16, 5)

print('Plotting Potentials...')

for ax_index, (var, ax) in enumerate(zip(recorded_vars, axs)):
  mean_exp = np.mean(np.array(trajectories[var]), axis=0)
  var_exp = np.var(np.array(trajectories[var]), axis=0)
  ax.plot(mean_exp, color='black', label='exp. mean')
  ax.fill_between(np.arange(200), mean_exp-var_exp, mean_exp+var_exp, facecolor='black', alpha=0.2, label='exp. var')
  ax.plot(mean_exp + var_th,  color='black', linestyle='dashed', label='th. var')
  ax.plot(mean_exp - var_th, color='black', linestyle='dashed')
  ax.set_xlabel(r"$t$ (ms)", fontsize=20)
  ax.tick_params(axis='both', labelsize=18)
  ax.set_title(var, fontsize=18, pad=5)
  if ax_index == 0:
    ax.set_ylabel(r"$u$ (mV)", fontsize=20)
    ax.legend(frameon=False, fontsize=18)

print('Setting Input Specs')

sns.despine()

input_specs["rate_on"] = 10*Hz
input_specs["rate_off"] = 1*Hz

exam_specs = {}
exam_specs["constant_list"] = ["none"]
exam_specs["batch_size"] = 16
exam_specs["recorded_vars"] = ["curr_conv1", "spk_conv1", "curr_conv2", "spk_conv2", "curr_total", "spk_rec", "curr_latent", "spk_latent"]
exam_specs["path"] = 'data/content/exam_world_data_1_1'
exams_dict, av_recorded_dict = get_exam_per_constant(network, input_specs, label_specs, exam_specs, device)
recordings = av_recorded_dict["none"]

layers_w = [to_np(param).flatten() for name, param in network.named_parameters()]
names_w = [name for name, param in network.named_parameters()]

curr_vars = ["curr_conv1", "curr_conv2", "curr_total", "curr_latent"]
layers_pA = [curr_to_pA(torch.mean(recordings[curr_var], 0), network).flatten()/pA for curr_var in curr_vars]

spk_vars = ["spk_conv1", "spk_conv2", "spk_rec", "spk_latent"]
layers_fr = [get_fr(recordings[spk_var], network).flatten()/Hz for spk_var in spk_vars]

netp, op = network.network_params, network.oscillation_params

A = 1/np.sqrt(1 + (netp["tau_m"]*op["omega"])**2)
expon = exp(-op["T"]/netp["tau_m"])
frac_num = (netp["v_th"])
frac_denom = (1 - expon)*netp["R_m"]*op["I_osc"]*A

I_min = (frac_num/frac_denom - 1)*op["I_osc"]*A
I_max = (frac_num/frac_denom + 1)*op["I_osc"]*A

print('Plotting Biophys. Stats...')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax2.sharey(ax3)

fig.suptitle('Biophys. Statistics', fontsize=20)

ax1.set_ylabel("Parameters", fontsize=20)
ax1.violinplot(layers_w, vert=False)
ax1.set_yticks(np.arange(1, len(names_w) +1))
ax1.set_yticklabels(names_w, fontsize=16)
ax1.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
ax1.tick_params(axis='both', labelsize=16)
ax1.set_xlabel("Efficacy (a.u.)", fontsize=20)

ax2.violinplot(layers_pA, vert=False)
ax2.tick_params(axis='both', labelsize=16)
ax2.set_yticks([1, 2, 3, 4, 5])
ax2.set_yticklabels( ["conv1", "conv2", "rec", "latent", "test"])
ax2.set_ylabel("Layer", fontsize=20)
ax2.set_xlabel("Input (pA)", fontsize=20)
ax2.axvline(x=I_min/pA, color='red', linestyle='dashed')
ax2.axvline(x=I_max/pA, color='red', linestyle='dashed')
ax2.text(I_min/pA-450, 0.65, "sub-threshold", fontsize=16, color='red')
ax2.text(1.1*I_max/pA, 0.65, "supra-locking", fontsize=16, color='red')

ax3.violinplot(layers_fr, vert=False)
ax3.set_xticks([0, 50, 100, 150, 200])
ax3.set_xticklabels([0, 50, 100, 150, 200], fontsize=20)
ax3.tick_params(left = False, labelleft=False, labelsize=16)
ax3.set_xlabel("FR (Hz)", fontsize=20)


fig.set_size_inches(16, 5)

sns.despine()

# what is this recorded dict?
#plt.imshow(torch.mean(recorded["curr_conv1"], 0)[2, 11].detach().cpu().numpy())

plt.imshow(torch.mean(network.conv1.weight[11], 1).detach().cpu().numpy())
plt.set_cmap('gray_r')
plt.colorbar()

torch.mean(network.conv1.weight, (2, 3)).shape

plt.plot(np.transpose(torch.mean(network.conv1.weight, (2, 3)).detach().cpu().numpy()))

network = Net(time_params, network_params, oscillation_params, frame_params, convolution_params, device).to(device)

print('Testing Cogntiive Performance...')
"""## Test network cognitive performance"""
exam_specs = {}
exam_specs["constant_list"] = ["none", "triangle", "square", "circle"]
exam_specs["batch_size"] = 64
exam_specs["recorded_vars"] = ["curr_rec"]
exam_specs["path"] = 'data/content/exam_world_data_1_1'

exams_dict, av_recorded_dict = get_exam_per_constant(network, input_specs, label_specs, exam_specs, device)

exams_dict["none"]

delta_curr_out = (torch.mean(av_recorded_dict["triangle"]["curr_rec"], 0) - torch.mean(av_recorded_dict["none"]["curr_rec"], 0))/torch.mean(av_recorded_dict["none"]["curr_rec"].flatten())

plt.hist(delta_curr_out.detach().cpu().flatten(),)

