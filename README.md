# MSc_Project - Characterisation of a Convolutional, Spiking Autoencoder
 
This code comprises the practical element of my MSc Thesis titled: A Characterisation of Convolutional Spiking Autoencoders.

.csv files in the main directory are used to generate plots in the plotting.py and plotting_freqs.py files.

The remaining files are a mixture of imports for the code in the main.py file, and supplementary visualisation tool for various datasets including MNIST, DVS, and SHD. The main.py file is configured to work with Weights and Biases (wandb), a cloud-based machine learning experiment platform, and will require reconfiguration to run locally.

A brief summary of the main files is provided as follows:

1. main.py - integrates imports, training/testing loop, and plotting
2. image_to_image.py - defining the network
3. train_network.py - contains training function for spiking and non-spiking networks
4. helpers.py - contains most of the helper functions for conversion, training, and dataloading

Other files are included but are supplementary and don't impact the main spiking network under analysis. Further information on these can be provided on request.

Any questions, I'd be happy to address them at luke.alderson23@alumni.imperial.ac.uk.
