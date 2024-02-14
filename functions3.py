# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:20:51 2024

@author: Hassan Rezvan

This file includes all functions implemented in Ex_3.py file
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, tree
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def normalizing(band):
    """
    normalize band.
    
    Arguments:
    band: band to be normalized
    
    Returns: normalized band
    """
    return (band-np.amin(band))/(np.amax(band)-np.amin(band))


def Accuracy(Confusion_Matrix):
    """
    calculates accuracy measures.
    
    Arguments:
    Confusion_Matrix
    
    Returns: 
    OA: Overall Accuracy
    CA: Producer's Accuracy
    UA: User's Accuracy
    """
    
    OA = np.trace(Confusion_Matrix)/np.sum(Confusion_Matrix)
    PA = []
    UA = []
    for i in range(5):
        PA.append(Confusion_Matrix[i,i]/np.sum(Confusion_Matrix[i,:]))
        UA.append(Confusion_Matrix[i,i]/np.sum(Confusion_Matrix[:,i]))
    return OA, PA, UA

def PlotMaps(map_name,description):
    """
    plots the input map.
    
    Arguments:
    map_name: the map to be displayed
    description: title of the map
    """
    plt.figure(figsize=(15,15),dpi=600)
    colors = ['red','green','blue','black','brown']
    cmap = ListedColormap(colors)
    legend_lebels = {'red':'Building','blue':'Water',
                      'black':'Road','green':'Vegetation','peachpuff':'Bare soil'}
    patches = [Patch(color=color,label=label) for color, label in legend_lebels.items()]
    plt.imshow(map_name,cmap=cmap)
    plt.title(description, fontsize=34)
    plt.xticks([]),plt.yticks([])
    plt.legend(handles=patches,bbox_to_anchor=(1.22,1),facecolor='white',
               prop={'size':20})

def Extract_trainingData(stack, Train_mask):
    """
    Extracts and splits training and test data.
    
    Arguments:
    stack: image to be classified
    Train_mask: Reshaped ground-Truth image
    
    Returns: 
    X_train: training data of X
    X_test: test data of X
    y_train: training data of y
    y_test: test data of y
    """
    stack_r = stack.reshape(stack.shape[0]*stack.shape[1], stack.shape[2])
    X = stack_r[np.where(Train_mask!=0)[0],:]
    Y = Train_mask[np.where(Train_mask!=0)]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, shuffle=True, train_size = .7)
    return X_train, X_test, y_train, y_test
    
def SVM_Classifier(img, X_train, y_train, X_test, kernel):
    """
    Classifies using SVM.
    
    Arguments:
    img: image to be classified
    X_train: training data, X label
    y_train: training data, y label
    X_test: test data, X
    kernel: type of kernel
    
    Returns: 
    Classified image
    Predicted test data
    """
    img_n_r = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    clf_svm = svm.SVC(C=30, kernel=kernel, degree=3, gamma=13, coef0=0, probability=True,
              cache_size=400, class_weight=None, decision_function_shape='ovr')
    clf_svm.fit(X_train, y_train)
    pred_1 = clf_svm.predict(img_n_r)
    pred_SVM = pred_1.reshape(img.shape[0],img.shape[1])
    pred_test_SVM = clf_svm.predict(X_test)
    return pred_SVM, pred_test_SVM

def DT_Classifier(img, X_train, y_train, X_test):
    """
    Classifies using Decision Tree.
    
    Arguments:
    img: image to be classified
    X_train: training data, X label
    y_train: training data, y label
    X_test: test data, X
    
    Returns: 
    Classified image
    Predicted test data
    """
    img_n_r = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    clf_tree = tree.DecisionTreeClassifier()
    clf_tree.fit(X_train, y_train)
    pred = clf_tree.predict(img_n_r)
    pred_DT = pred.reshape(img.shape[0],img.shape[1])
    pred_test_DT = clf_tree.predict(X_test)
    return pred_DT, pred_test_DT