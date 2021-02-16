# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:17:22 2020

@author: lenovo
"""

import scipy.io
import numpy as np

def create_sample(data_mat, time_lags):
    dim1, dim2 = data_mat.shape
    x = np.zeros((dim2 - np.max(time_lags), len(time_lags), dim1, 1))
    y = np.zeros((dim2 - np.max(time_lags), 1, dim1, 1))
    for i in range(dim2 - np.max(time_lags)):
        x[i, :, :, 0] = data_mat[:, i + time_lags - 1].T
        y[i, 0, :, 0] = data_mat[:, i + np.max(time_lags)]
    return x, y

time_lags = np.array([1, 2, 288])

pattern = ['PM0', 'PM0.1', 'PM0.2', 'PM0.4', 'CM0.1', 'CM0.2', 'CM0.4']

for pt in pattern:
    directory = '../../../datasets/Metr-LA-data-set/'
    A = np.load(directory + 'Metr_ADJ.npy')
    dense_mat = np.load( directory + 'Metr-LA.npy')
    print('Dataset shape:')
    print(dense_mat.shape)
    if pt[:2] == 'PM':
        missing_rate = float(pt[2:])
        print('Missing scenario: PM, %f'%(missing_rate))
        # =============================================================================
        ### Random missing (PM) scenario
        ### Set the RM scenario by:
        rm_random_mat = np.load(directory + 'rm_random_mat.npy')
        binary_mat = np.round(rm_random_mat + 0.5 - missing_rate)
    else:
        missing_rate = float(pt[2:])
        print('Missing scenario: CM, %f'%(missing_rate))
        # # =============================================================================
        ### Non random missing (CM) scenario
        # ### Set the RM scenario by:
        nm_random_mat = np.load(directory + 'nm_random_mat.npy')
        binary_tensor = np.zeros((dense_mat.shape[0], 61, 288))
        for i1 in range(binary_tensor.shape[0]):
            for i2 in range(binary_tensor.shape[1]):
                binary_tensor[i1, i2, :] = np.round(nm_random_mat[i1, i2] + 0.5 - missing_rate)
        binary_mat = binary_tensor.reshape([binary_tensor.shape[0], binary_tensor.shape[1] * binary_tensor.shape[2]])
        #================================================================================
    sparse_mat = np.multiply(dense_mat, binary_mat)

    test_rate = 0.082
    val_rate = 0.082 * 2

    test_len = int(sparse_mat.shape[1] * test_rate) + np.max(time_lags)
    val_len = int(sparse_mat.shape[1] * val_rate) + np.max(time_lags)
    train_len = sparse_mat.shape[1] - test_len - val_len

    training_set = sparse_mat[:, :train_len]
    val_set = sparse_mat[:, train_len: train_len + val_len]
    test_set = sparse_mat[:, train_len + val_len:]
    dense_test_set = dense_mat[:, train_len + val_len:]
    
    train_x, train_y = create_sample(training_set, time_lags)
    val_x, val_y = create_sample(val_set, time_lags)
    test_x, test_y = create_sample(test_set, time_lags)
    _, test_ground_truth = create_sample(dense_test_set, time_lags)

    print('Test sample shape:')
    print(test_x.shape)
    print('Test label shape:')
    print(test_y.shape)
    
    dir = '../data/metr_data/'
    outfile = dir + 'train' + pt + '.npz'
    np.savez(outfile, x=train_x, y=train_y)
    outfile = dir + 'val' + pt + '.npz'
    np.savez(outfile, x=val_x, y=train_y)
    outfile = dir + 'test' + pt + '.npz'
    np.savez(outfile, x=test_x, y=test_y, gdt=test_ground_truth)
    
    # Input (sample num, timesteps, num_sensor, input_dim)
    # Labels: ([sample num, timesteps, num_sensor, input_dim), same format with input except the temporal dimension.
    