# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 14:47:01 2023

@author: Hassan Rezvan

This file includes all functions implemented in Ex_1.py file
"""
import numpy as np
import matplotlib.pyplot as plt

def normalizing(band):
    """
    normalize band.
    
    Arguments:
    band: band to be normalized
    
    Returns: normalized band
    """
    return (band-np.amin(band))/(np.amax(band)-np.amin(band))


def shuffle_2D_matrix(matrix, axis = 0):
    """
    Shuffle 2D matrix by column or row.
    
    Arguments:
    matrix: 2D matrix to be shuffled
    axis  : zero - by column, non-zero - by row
    
    Returns:
    shuffled_matrix: shuffled matrix
    """
        
    if axis == 0:             # by column
        m = matrix.shape[1]
        permutation = list(np.random.permutation(m))
        shuffled_matrix = matrix[:, permutation]
    elif axis == 1:           # by row
        m = matrix.shape[0]
        permutation = list(np.random.permutation(m))
        shuffled_matrix = matrix[permutation, :]

    return shuffled_matrix


def train_test_split(data):
    """
    split training data to train and test data with the ratio of 70/30.
    
    Arguments:
    data: the whole extracted data in envi to be splited
    
    Returns:
    train_mat, test_mat: list of train data and test data for each class
    train_data, test_data: splited arrays incuding train and test data
    """
    train_mat = []
    test_mat = []
    for i in range(len(data)):
        matrix = data[i]
        index = int(len(matrix) * 0.7)
        train_mat.append(matrix[:index])
        test_mat.append(matrix[index:])
    train_data = np.vstack((train_mat[0],train_mat[1],train_mat[2],train_mat[3]))
    test_data = np.vstack((test_mat[0],test_mat[1],test_mat[2],test_mat[3]))
    
    return train_data, test_data, train_mat, test_mat
        
def dual_band_plot(img_reshaped):
    """
    plots bands to each other.
    
    Arguments:
    img_reshaped: image containing several bands
    """
    k=1
    plt.figure()
    for i in range(4):
        for j in range(4):
            if j>i:
                plt.subplot(3,3,k)
                plt.scatter(img_reshaped[:,i],img_reshaped[:,j])
                plt.xlabel("Band {}".format(i+1)), plt.ylabel("Band {}".format(j+1))
                plt.xticks([]),plt.yticks([])
                k = k+1
    
    

def band_mean_calculator(class_train):
    """
    calculates mean of each band.
    
    Arguments:
    class_train: collected train data for each class in Envi
    
    Returns:
    [mean bands]: a list containing band means
    """
    b1 = []
    b2 = []
    b3 = []
    b4 = []
    for i in range(len(class_train)):
        b1.append(class_train[i][0])
        b2.append(class_train[i][1])
        b3.append(class_train[i][2])
        b4.append(class_train[i][3])
        
    mean_b1 = np.mean(np.array(b1))
    mean_b2 = np.mean(np.array(b2))
    mean_b3 = np.mean(np.array(b3))
    mean_b4 = np.mean(np.array(b4))
        
    
    return [mean_b1, mean_b2, mean_b3, mean_b4]

    