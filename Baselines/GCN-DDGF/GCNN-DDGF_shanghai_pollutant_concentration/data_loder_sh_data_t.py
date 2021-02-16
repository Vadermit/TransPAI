# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:17:22 2020

@author: lenovo
"""

import scipy.io
import numpy as np

def create_sample(data_mat, time_lag):
    dim1, dim2 = data_mat.shape
    x = np.zeros((dim2 - time_lag, time_lag, dim1, 1))
    y = np.zeros((dim2 - time_lag, 1, dim1, 1))
    for i in range(dim2 - time_lag):
        x[i, :, :, 0] = data_mat[:, i : i + time_lag].T
        y[i, 0, :, 0] = data_mat[:, i + time_lag]
    return x, y

time_lag = 24

pattern = ['PM0', 'PM0.1', 'PM0.2', 'PM0.4', 'CM0.1', 'CM0.2', 'CM0.4']
for pt in pattern:
    tensor = scipy.io.loadmat('C:/Users/lenovo/Shanghai-traffic-station-pollutant-data-set/NTS_tensor.mat')
    dense_tensor = tensor['tensor']
    print('The shape of the initial dataset is:')
    print(dense_tensor.shape)
    dim1, dim2, dim3 = dense_tensor.shape
    if pt[:2] == 'RM':
        missing_rate = float(pt[2:])
        print('Missing scenario: RM, %f'%(missing_rate))
        # =============================================================================
        ### Random missing (PM) scenario
        ### Set the RM scenario by:
        tensor = scipy.io.loadmat('C:/Users/lenovo/Shanghai-traffic-station-pollutant-data-set/NTS_random_tensor.mat')
        random_tensor = tensor['tensor']
        binary_tensor = np.ones((dim1, dim2, dim3))
        binary_tensor[random_tensor < missing_rate] = 0 
    else:
        missing_rate = float(pt[2:])
        print('Missing scenario: PM, %f'%(missing_rate))
        # # =============================================================================
        ### Non random missing (CM) scenario
        # ### Set the RM scenario by:
        missing_period = 6 #data missing in continuous [6, 12, 24, 48] hours
        random_array_file = 'C:/Users/lenovo/Shanghai-traffic-station-pollutant-data-set/NTS_random_array' + str(missing_period) + '.mat'
        tensor = scipy.io.loadmat(random_array_file)
        random_array = tensor['array'][0]
        binary_reshape_tensor = np.ones_like(dense_tensor)
        binary_reshape_tensor = binary_reshape_tensor.reshape(dim1,dim2,int(dim3 / missing_period),missing_period)
        pos = np.where(random_array < missing_rate)
        binary_reshape_tensor[:, :, pos, :] = 0
        binary_tensor = binary_reshape_tensor.reshape(dim1, dim2, dim3)
    sparse_tensor = np.multiply(dense_tensor, binary_tensor)
    dense_tensor = dense_tensor[:, :, 5768:]
    sparse_tensor = sparse_tensor[:, :, 5768:]
    dim1, dim2, dim3 = sparse_tensor.shape
    dense_mat = dense_tensor.reshape([dim1*dim2, dim3])
    sparse_mat = sparse_tensor.reshape([dim1*dim2, dim3])

    test_rate = 0.082
    val_rate = 0.1
    train_val_len = int((1 - test_rate) * sparse_mat.shape[1] - time_lag)
    test_len = sparse_mat.shape[1] - train_val_len
    train_len = int((1 - val_rate) * train_val_len)
    
    training_set = sparse_mat[:, :train_len]
    val_set = sparse_mat[:, train_len: train_val_len]
    test_set = sparse_mat[:, train_val_len:]
    dense_test_set = dense_mat[:, train_val_len:]
    
    train_x, train_y = create_sample(training_set, time_lag)
    val_x, val_y = create_sample(val_set, time_lag)
    test_x, test_y = create_sample(test_set, time_lag)
    _, test_ground_truth = create_sample(dense_test_set, time_lag)
    dir = '../data/shanghai_data/'
    outfile = dir + 'train' + pt + '.npz'
    np.savez(outfile, x=train_x, y=train_y)
    outfile = dir + 'val' + pt + '.npz'
    np.savez(outfile, x=val_x, y=train_y)
    outfile = dir + 'test' + pt + '.npz'
    np.savez(outfile, x=test_x, y=test_y, gdt=test_ground_truth)
    
    # Input (sample num, timesteps, num_sensor, input_dim)
    # Labels: ([sample num, timesteps, num_sensor, input_dim), same format with input except the temporal dimension.
    