# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:19:37 2024

@author: Hassan Rezvan

The main file to run
"""

# Library imports
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# _____________________________________________
# Function imports
from functions import normalizing
from functions import Accuracy
from functions import PlotMaps
from functions import Extract_trainingData
from functions import SVM_Classifier
from functions import DT_Classifier
# _____________________________________________
img = imread('subset.tif')
plt.rcParams['font.family'] = 'Times New Roman'


texture = imread('texture_img.tif')
texture_mean = texture[0,:,:]
texture_variance = texture[1,:,:]
texture_homogeneity = texture[2,:,:]
texture_contrast = texture[3,:,:]
texture_disimilarity = texture[4,:,:]

img_n = normalizing(img)
red_normalized = img_n[:,:,0]
green_normalized = img_n[:,:,1]
blue_normalized = img_n[:,:,2]
nir_normalized = img_n[:,:,3]
b1 = img[:,:,0]  # RED
b2 = img[:,:,1]  # GREEN
b3 = img[:,:,2]  # BLUE
b4 = img[:,:,3]  # NIR

#_________________________________________________________
## Defining indexes
CIg = (b4/b2)-1
PISI = 0.8192*blue_normalized-0.5735*nir_normalized+0.0750
EVI = 2.5*((nir_normalized-red_normalized)/(nir_normalized+6*red_normalized-7.5*blue_normalized+1))

#_________________________________________________________
## Stacking
stack_1 = np.stack((blue_normalized,green_normalized,red_normalized),axis=2)
stack_2 = np.stack((blue_normalized,green_normalized,red_normalized,CIg,PISI),axis=2)
stack_3 = np.stack((blue_normalized,green_normalized,red_normalized,CIg,PISI,EVI,texture_mean,texture_variance,texture_homogeneity,texture_contrast,texture_disimilarity),axis=2)

#_________________________________________________________
# Ground truth data
GT = imread('GT.tif')
legend_lab = ['Building', 'Vegetation', 'Water', 'Road', 'Soil']
#_________________________________________________________
# Train-Test splitting
img_n_r = img_n.reshape(img_n.shape[0]*img_n.shape[1], 4)
Train_mask = GT.reshape(-1)
X_train1, X_test1, y_train1, y_test1 = Extract_trainingData(stack_1, Train_mask)
X_train2, X_test2, y_train2, y_test2 = Extract_trainingData(stack_2, Train_mask)
X_train3, X_test3, y_train3, y_test3 = Extract_trainingData(stack_3, Train_mask)

#_________________________________________________________
# SVM Classification
time_p_i = time.time()
pred_SVM_1, pred_test_SVM_1 = SVM_Classifier(stack_1, X_train1, y_train1, X_test1, 'poly')
PlotMaps(pred_SVM_1,"SVM classification | Stack 1")
cm_svm1 = confusion_matrix(y_test1, pred_test_SVM_1)
time_p_f = time.time()
print('Time spent for SVM classification of Stack_1: ', time_p_f-time_p_i)
OA_SVM_1, UA_SVM_1, PA_SVM_1 = Accuracy(cm_svm1)
ConfusionMatrixDisplay(confusion_matrix=cm_svm1, display_labels=legend_lab).plot()

time_p_i = time.time()
pred_SVM_2, pred_test_SVM_2 = SVM_Classifier(stack_2, X_train2, y_train2, X_test2, 'poly')
PlotMaps(pred_SVM_2,"SVM classification | Stack 2")
cm_svm2 = confusion_matrix(y_test2, pred_test_SVM_2)
time_p_f = time.time()
print('Time spent for SVM classification of Stack_2: ', time_p_f-time_p_i)
OA_SVM_2, UA_SVM_2, PA_SVM_2 = Accuracy(cm_svm2)
ConfusionMatrixDisplay(confusion_matrix=cm_svm2, display_labels=legend_lab).plot()

time_p_i = time.time()
pred_SVM_3, pred_test_SVM_3 = SVM_Classifier(stack_3, X_train3, y_train3, X_test3, 'poly')
PlotMaps(pred_SVM_3,"SVM classification | Stack 3")
cm_svm3 = confusion_matrix(y_test3, pred_test_SVM_3)
time_p_f = time.time()
print('Time spent for SVM classification of Stack_3: ', time_p_f-time_p_i)
OA_SVM_3, UA_SVM_3, PA_SVM_3 = Accuracy(cm_svm3)
ConfusionMatrixDisplay(confusion_matrix=cm_svm3, display_labels=legend_lab).plot()

#_________________________________________________________
# Decision Tree Classification
time_p_i = time.time()
pred_DT_1, pred_test_DT_1 = DT_Classifier(stack_1, X_train1, y_train1, X_test1)
PlotMaps(pred_DT_1,"Decision Tree classification | Stack 1")
cm_dt1 = confusion_matrix(y_test1, pred_test_DT_1)
time_p_f = time.time()
print('Time spent for DT classification of Stack_1: ', time_p_f-time_p_i)
OA_DT_1, UA_DT_1, PA_DT_1 = Accuracy(cm_dt1)
ConfusionMatrixDisplay(confusion_matrix=cm_dt1, display_labels=legend_lab).plot()

time_p_i = time.time()
pred_DT_2, pred_test_DT_2 = DT_Classifier(stack_2, X_train2, y_train2, X_test2)
PlotMaps(pred_DT_2,"Decision Tree classification | Stack 2")
cm_dt2 = confusion_matrix(y_test2, pred_test_DT_2)
time_p_f = time.time()
print('Time spent for DT classification of Stack_2: ', time_p_f-time_p_i)
OA_DT_2, UA_DT_2, PA_DT_2 = Accuracy(cm_dt2)
ConfusionMatrixDisplay(confusion_matrix=cm_dt2, display_labels=legend_lab).plot()

time_p_i = time.time()
pred_DT_3, pred_test_DT_3 = DT_Classifier(stack_3, X_train3, y_train3, X_test3)
PlotMaps(pred_DT_3,"Decision Tree classification | Stack 3")
cm_dt3 = confusion_matrix(y_test3, pred_test_DT_3)
time_p_f = time.time()
print('Time spent for DT classification of Stack_3: ', time_p_f-time_p_i)
OA_DT_3, UA_DT_3, PA_DT_3 = Accuracy(cm_dt3)
ConfusionMatrixDisplay(confusion_matrix=cm_dt3, display_labels=legend_lab).plot()

#_________________________________________________________
# Kernel-based SVM Classification
time_p_i = time.time()
pred_SVM_1_linear, pred_test_SVM_1_linear = SVM_Classifier(stack_1, X_train1, y_train1, X_test1, 'linear')
PlotMaps(pred_SVM_1_linear,"SVM classification | Stack 1 | Linear kernel")
cm_svm1_linear = confusion_matrix(y_test1, pred_test_SVM_1_linear)
time_p_f = time.time()
print('Time spent for Linear-SVM classification of Stack_1: ', time_p_f-time_p_i)
OA_SVM_1_linear, UA_SVM_1_linear, PA_SVM_1_linear = Accuracy(cm_svm1_linear)
ConfusionMatrixDisplay(confusion_matrix=cm_svm1_linear, display_labels=legend_lab).plot()

time_p_i = time.time()
pred_SVM_1_rbf, pred_test_SVM_1_rbf = SVM_Classifier(stack_1, X_train1, y_train1, X_test1, 'rbf')
PlotMaps(pred_SVM_1_rbf,"SVM classification | Stack 1 | RBF kernel")
cm_svm1_rbf = confusion_matrix(y_test1, pred_test_SVM_1_rbf)
time_p_f = time.time()
print('Time spent for RBF-SVM classification of Stack_1: ', time_p_f-time_p_i)
OA_SVM_1_rbf, UA_SVM_1_rbf, PA_SVM_1_rbf = Accuracy(cm_svm1_rbf)
ConfusionMatrixDisplay(confusion_matrix=cm_svm1_rbf, display_labels=legend_lab).plot()

time_p_i = time.time()
pred_SVM_1_sigmoid, pred_test_SVM_1_sigmoid = SVM_Classifier(stack_1, X_train1, y_train1, X_test1, 'sigmoid')
PlotMaps(pred_SVM_1_sigmoid,"SVM classification | Stack 1 | Sigmoid kernel")
cm_svm1_sigmoid = confusion_matrix(y_test1, pred_test_SVM_1_sigmoid)
time_p_f = time.time()
print('Time spent for Sigmoid-SVM classification of Stack_1: ', time_p_f-time_p_i)
OA_SVM_1_sigmoid, UA_SVM_1_sigmoid, PA_SVM_1_sigmoid = Accuracy(cm_svm1_sigmoid)
ConfusionMatrixDisplay(confusion_matrix=cm_svm1_sigmoid, display_labels=legend_lab).plot()

