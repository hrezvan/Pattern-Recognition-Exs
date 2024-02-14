# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 21:59:50 2023

@author: Hassan Rezvan
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
    train_data = np.vstack((train_mat[0],train_mat[1],train_mat[2],train_mat[3],train_mat[4]))
    test_data = np.vstack((test_mat[0],test_mat[1],test_mat[2],test_mat[3],test_mat[4]))
    
    m = train_data.shape[1]
    train_data = np.delete(train_data, [m-2, m-1], axis=1)
    test_data = np.delete(test_data, [m-2, m-1], axis=1)
    
    test_rows_columns = [[] for _ in range(5)]
    for i in range(5):
        test_rows_columns[i] = test_mat[i][:,m-2:]
    for i in range(5):
        train_mat[i] = np.delete(train_mat[i], [m-2, m-1], axis=1)
        test_mat[i] = np.delete(test_mat[i], [m-2, m-1], axis=1)
    
    return train_data, test_data, train_mat, test_mat, test_rows_columns 


def band_mean_calculator(class_train,n):
    """
    calculates mean of each band.
    
    Arguments:
    class_train: collected train data for each class in Envi
    n: number of bands
    
    Returns:
    [mean bands]: a list containing band means
    """
    bands = [[] for _ in range(n)]
    
    for i in range(len(class_train)):
        for j in range(n):
            bands[j].append(class_train[i][j])
    
    # Calculate the mean for each band
    mean_bands = [np.mean(np.array(band)) for band in bands]
    
    return mean_bands


def Bayes_DF(img,mean,cov,P_w=1):
    """
    classifies the image based on Bayes theory.
    
    Arguments:
    img: image to be classified
    mean: a list of band means for each class
    cov: a list of covariance matrix for each class
    P_w: a list acontaining prior probability of each class(default = 0)
    
    Returns: 
    g: Probability membership function for a class.
    """
    
    inv_cov = np.linalg.inv(cov)
    ln_det_cov = np.log(np.linalg.det(cov))
    g = np.zeros((img.shape[0],img.shape[1]))
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            q = img[i,j,:] - mean
            x = (0.5*((q.T).dot(inv_cov)))
            g[i,j] = -((x.dot(q))) - (0.5*(ln_det_cov)) + np.log(P_w)
    return g


def Accuracy(Confusion_Matrix):
    """
    calculates accuracy measures.
    
    Arguments:
    Confusion_Matrix
    
    Returns: 
    OA: Overall Accuracy
    CA: Class Accuracy
    """
    
    OA = np.trace(Confusion_Matrix)/np.sum(Confusion_Matrix)
    CA = []
    for i in range(5):
        CA.append(Confusion_Matrix[i,i]/np.sum(Confusion_Matrix[i,:]))
        
    return OA,CA

def PlotMaps(map_name,labels,cmap,patches,description):
    """
    plots the input map.
    
    Arguments:
    map_name: the map to be displayed
    labels
    cmap
    patches
    description: title of the map
    """
    classification_map_labels = np.array(labels)[map_name]
    plt.figure(figsize=(15,20))
    plt.imshow(map_name, cmap=cmap)
    plt.legend(handles=patches,bbox_to_anchor=(1.22,1),facecolor='white',
                prop={'size':20})
    plt.title(description, fontsize=34)
    plt.xticks([]),plt.yticks([])
    plt.show()
    
    