import os
import h5py
import numpy as np

from .frames_generator import latent_generator, latent_to_frame, label_from_matrix


def make_dataset(train_samples, test_samples, width, M_min, M_max, restrictions, path):
    
    latent_dim = 8
    label_size = latent_dim*M_max

    train_array = np.zeros((train_samples, 3, width, width))
    train_labels = np.zeros((train_samples, label_size))
    test_array = np.zeros((test_samples, 3, width, width))
    test_labels = np.zeros((test_samples, label_size))

    for i in range(train_samples):
        l_matrix = latent_generator(M_min, M_max, width, restrictions)
        frame = latent_to_frame(l_matrix, width)
        label = label_from_matrix(l_matrix, M_max, width)
        train_array[i] = frame
        train_labels[i] = label

    for i in range(test_samples):
        l_matrix = latent_generator(M_min, M_max, width, restrictions)
        frame = latent_to_frame(l_matrix, width)
        label = label_from_matrix(l_matrix, M_max, width)
        test_array[i] = frame
        test_labels[i] = label
        
    dir_path = '{}/world_data_{}_{}'.format(path, M_min, M_max)
    
    os.mkdir(dir_path)

    with h5py.File(dir_path+'/train.hdf5','w') as f:
        group = f.create_group('data')
        group.create_dataset(name='frames', data=train_array, chunks=True, compression='gzip')
        group.create_dataset(name='labels', data=train_labels, chunks=True, compression='gzip')

    with h5py.File(dir_path+'/test.hdf5','w') as f:
        group = f.create_group('data')
        group.create_dataset(name='frames', data=test_array, chunks=True, compression='gzip')
        group.create_dataset(name='labels', data=test_labels, chunks=True, compression='gzip')
        
        
        
def make_exam_tests(test_samples, width, M_min, M_max, path):
    
    latent_dim = 8
    label_size = latent_dim*M_max
    
    constant_list = ["none", "north_west", "north_center", "north_east", "center_west", "center_center", "center_east", "south_west", "south_center", "south_east", "red", "green", "blue", "triangle", "square", "circle"]
    restriction_list = [{}, {"position": (0, 28)}, {"position": (14, 28)}, {"position": (28, 28)}, {"position": (0, 14)}, {"position": (14, 14)}, {"position": (28, 14)}, {"position": (0, 0)}, {"position": (14, 0)}, {"position": (28, 0)}, {"color": (0, 0, 1)}, {"color": (0, 1, 0)}, {"color": (1, 0, 0)}, {"shape":"triangle"}, {"shape":"square"}, {"shape":"circle"}]
    
    dir_path = '{}/exam_world_data_{}_{}'.format(path, M_min, M_max) 
    os.mkdir(dir_path)
    
    for constant, restrictions in zip(constant_list, restriction_list):

        test_array = np.zeros((test_samples, 3, width, width))
        test_labels = np.zeros((test_samples, label_size))

        for i in range(test_samples):
            l_matrix = latent_generator(M_min, M_max, width, restrictions)
            frame = latent_to_frame(l_matrix, width)
            label = label_from_matrix(l_matrix, M_max, width)
            test_array[i] = frame
            test_labels[i] = label
            
        with h5py.File(dir_path+'/test_{}.hdf5'.format(constant),'w') as f:
                group = f.create_group('data')
                group.create_dataset(name='frames', data=test_array, chunks=True, compression='gzip')
                group.create_dataset(name='labels', data=test_labels, chunks=True, compression='gzip')