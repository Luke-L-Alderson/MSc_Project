import tonic
import torch
import tonic.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tonic import DiskCachedDataset
from torchvision.transforms import v2
from matplotlib import pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML
def build_nmnist_dataset(train_specs, input_specs = None):
    
    batch_size = train_specs["batch_size"]
    sensor_size = tonic.datasets.NMNIST.sensor_size
    
    
    raw_transform = tonic.transforms.Compose([
                transforms.Denoise(filter_time=10000),
                transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
                ])

    cache_transform = tonic.transforms.Compose([
                #transforms.Denoise(filter_time=10000),
                #transforms.ToFrame(sensor_size=sensor_size, time_window=1000),
                #v2.Grayscale(),
                torch.from_numpy,
                #v2.ToTensor(),
                #v2.Normalize((0,), (1,)),
                ])
    
    train_dataset = tonic.datasets.NMNIST(save_to='./dataset',
                                          transform=raw_transform,
                                          train=True)
    
    test_dataset = tonic.datasets.NMNIST(save_to='./dataset',
                                          transform=raw_transform,
                                          train=False)
    
    #events, target = train_dataset[0]
    
    
    train_dataset = DiskCachedDataset(train_dataset, cache_path='./cache/nmnist/train', transform=cache_transform)
    test_dataset = DiskCachedDataset(test_dataset, cache_path='./cache/nmnist/train')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))
    
    event_tensor, target = next(iter(train_loader))
    print(f"Data in form: {event_tensor.shape}")
    
    return train_dataset, train_loader, test_dataset, test_loader

train_specs = {}
train_specs["batch_size"] = 64

train_dataset, train_loader, test_dataset, test_loader = build_nmnist_dataset(train_specs)

print(type(train_dataset))

event_tensor, target = next(iter(train_loader))

print(event_tensor.shape)
print(target.shape)
sample_index = 0

for i in range(2):
    print(target[sample_index])
    fig1, ax1 = plt.subplots()
    animrec = splt.animator(event_tensor[:, sample_index, i], fig1, ax1)
    HTML(animrec.to_html5_video())
    plt.show()
    animrec.save(f"figures/spike_nmnist_{i}.gif")