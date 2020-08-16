# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 19:39:45 2020

@author: lenovo
"""
import pandas as pd
import numpy as np

speed_matrix =  pd.read_pickle('speed_matrix_2015')
dense_mat = speed_matrix.values.T
dense_mat = dense_mat[:, - 288 * 61:]

dim1, dim2 = dense_mat.shape
print('Shape of the dataset:')
print(dense_mat.shape)

np.random.seed(67)

np.save('dense_mat.npy', dense_mat)

# Create RM random matrix
random_mat = np.random.rand(dim1, dim2)
np.save('rm_random_mat.npy', random_mat)

# Create NM random matrix
dense_mat_reshape = dense_mat.reshape((dim1, 61, int(dim2 / 61)))
random_mat = np.random.rand(dim1, 61)
np.save('nm_random_mat.npy', random_mat)

'''
A = np.array([[1, 2, 3, 4, 5, 6],[7, 8, 9, 10, 11, 12]])
B = A.reshape((2, 3, 2))
C = B.reshape((2, 6))
print(A)
print()
print(B)
print()
print(C)
'''