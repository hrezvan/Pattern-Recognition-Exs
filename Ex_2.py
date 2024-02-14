# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 21:58:14 2023

@author: Hassan Rezvan
"""

# Library imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from skimage.io import imread
import time
# _____________________________________________
# Function imports
from functions import normalizing
from functions import train_test_split
from functions import shuffle_2D_matrix
from functions import band_mean_calculator
from functions import Bayes_DF
from functions import Accuracy
from functions import PlotMaps
# _____________________________________________
img = imread('subset.tif')
plt.rcParams['font.family'] = 'Times New Roman'


texture = imread('texture_img.tif')
texture_mean = texture[0,:,:]
texture_variance = texture[1,:,:]
texture_homogeneity = texture[2,:,:]
texture_contrast = texture[3,:,:]
texture_disimilarity = texture[4,:,:]

b1 = img[:,:,0]  # RED
b2 = img[:,:,1]  # GREEN
b3 = img[:,:,2]  # BLUE
b4 = img[:,:,3]  # NIR
red_normalized = normalizing(b1)
green_normalized = normalizing(b2)
blue_normalized = normalizing(b3)
nir_normalized = normalizing(b4)

#_________________________________________________________
## Defining indexes
CIg = (b4/b2)-1
PISI = 0.8192*blue_normalized-0.5735*nir_normalized+0.0750
EVI = 2.5*((nir_normalized-red_normalized)/(nir_normalized+6*red_normalized-7.5*blue_normalized+1))

#_________________________________________________________
## Stacking
stack_1 = np.stack((b3,b2,b1),axis=2)
stack_2 = np.stack((b3,b2,b1,CIg,PISI),axis=2)
stack_3 = np.stack((b3,b2,b1,CIg,PISI,EVI,texture_mean,texture_variance,texture_homogeneity,texture_contrast,texture_disimilarity),axis=2)

#_________________________________________________________
# Ground truth data
GT = imread('GT.tif')
colors = ['white','red','green','blue','black','brown']
cmap = ListedColormap(colors)
legend_lebels = {'white':'background','red':'Building','blue':'Water',
                  'black':'Road','green':'Vegetation','peachpuff':'Bare soil'}
patches = [Patch(color=color,label=label) for color, label in legend_lebels.items()]
        
#______________________________________
# Stack 1
train_building = stack_1[GT==1]
train_building = np.hstack((train_building,np.array(np.where(GT == 1)).T))
train_veg = stack_1[GT==2]
train_veg = np.hstack((train_veg,np.array(np.where(GT == 2)).T))
train_water = stack_1[GT==3]
train_water = np.hstack((train_water,np.array(np.where(GT == 3)).T))
train_road = stack_1[GT==4]
train_road = np.hstack((train_road,np.array(np.where(GT == 4)).T))
train_soil = stack_1[GT==5]
train_soil = np.hstack((train_soil,np.array(np.where(GT == 5)).T))
whole_training_data = []
whole_training_data.append(train_building)
whole_training_data.append(train_veg)
whole_training_data.append(train_water)
whole_training_data.append(train_road)
whole_training_data.append(train_soil)

shuffled = []
for i in range(len(whole_training_data)):
    shuffled.append(shuffle_2D_matrix(whole_training_data[i], axis = 1))

train_data, test_data, train_list, test_list, test1_rows_columns = train_test_split(shuffled)

mean_c1 = band_mean_calculator(train_list[0],3)
mean_c2 = band_mean_calculator(train_list[1],3)
mean_c3 = band_mean_calculator(train_list[2],3)
mean_c4 = band_mean_calculator(train_list[3],3)
mean_c5 = band_mean_calculator(train_list[4],3)
Mean1 = np.array([mean_c1,mean_c2,mean_c3,mean_c4,mean_c5])

cov_c1 = np.cov(train_list[0].reshape(3,len(train_list[0])))
cov_c2 = np.cov(train_list[1].reshape(3,len(train_list[1])))
cov_c3 = np.cov(train_list[2].reshape(3,len(train_list[2])))
cov_c4 = np.cov(train_list[3].reshape(3,len(train_list[3])))
cov_c5 = np.cov(train_list[4].reshape(3,len(train_list[4])))
Cov1 = np.array((cov_c1,cov_c2,cov_c3,cov_c4,cov_c5))

time_1_i = time.time()
G1 = np.zeros((img.shape[0],img.shape[1],5))
for n in range(5):       # number of classes
    G1[:,:,n] = Bayes_DF(stack_1,Mean1[n,:],Cov1[n,:,:])

ML1 = np.argmax(G1.reshape(img.shape[0]*img.shape[1],5),axis = 1)
time_1_f = time.time()
print('Time spent for stack_1: ', time_1_f-time_1_i)
ML1 = ML1.reshape(img.shape[0],img.shape[1])
# List of labels
labels = ['Building', 'Vegetation', 'Water', 'Road', 'Soil']
cmap = ListedColormap(['red','green','blue','black','peachpuff'])
PlotMaps(ML1,labels,cmap,patches,'Stack_1 map')

#______________________________________
# Stack 2
train_building = stack_2[GT==1]
train_building = np.hstack((train_building,np.array(np.where(GT == 1)).T))
train_veg = stack_2[GT==2]
train_veg = np.hstack((train_veg,np.array(np.where(GT == 2)).T))
train_water = stack_2[GT==3]
train_water = np.hstack((train_water,np.array(np.where(GT == 3)).T))
train_road = stack_2[GT==4]
train_road = np.hstack((train_road,np.array(np.where(GT == 4)).T))
train_soil = stack_2[GT==5]
train_soil = np.hstack((train_soil,np.array(np.where(GT == 5)).T))
whole_training_data = []
whole_training_data.append(train_building)
whole_training_data.append(train_veg)
whole_training_data.append(train_water)
whole_training_data.append(train_road)
whole_training_data.append(train_soil)

shuffled = []
for i in range(len(whole_training_data)):
    shuffled.append(shuffle_2D_matrix(whole_training_data[i], axis = 1))

train_data, test_data, train_list, test_list, test2_rows_columns = train_test_split(shuffled)

mean_c1 = band_mean_calculator(train_list[0],5)
mean_c2 = band_mean_calculator(train_list[1],5)
mean_c3 = band_mean_calculator(train_list[2],5)
mean_c4 = band_mean_calculator(train_list[3],5)
mean_c5 = band_mean_calculator(train_list[4],5)
Mean2 = np.array([mean_c1,mean_c2,mean_c3,mean_c4,mean_c5])

cov_c1 = np.cov(train_list[0].reshape(5,len(train_list[0])))
cov_c2 = np.cov(train_list[1].reshape(5,len(train_list[1])))
cov_c3 = np.cov(train_list[2].reshape(5,len(train_list[2])))
cov_c4 = np.cov(train_list[3].reshape(5,len(train_list[3])))
cov_c5 = np.cov(train_list[4].reshape(5,len(train_list[4])))
Cov2 = np.array((cov_c1,cov_c2,cov_c3,cov_c4,cov_c5))

time_2_i = time.time()
G2 = np.zeros((img.shape[0],img.shape[1],5))
for n in range(5):       # number of classes
    G2[:,:,n] = Bayes_DF(stack_2,Mean2[n,:],Cov2[n,:,:])

ML2 = np.argmax(G2.reshape(img.shape[0]*img.shape[1],5),axis = 1)
time_2_f = time.time()
print('Time spent for stack_2: ', time_2_f-time_2_i)
ML2 = ML2.reshape(img.shape[0],img.shape[1])
PlotMaps(ML2,labels,cmap,patches,'Stack_2 map')

#______________________________________
# Stack 3
train_building = stack_3[GT==1]
train_building = np.hstack((train_building,np.array(np.where(GT == 1)).T))
train_veg = stack_3[GT==2]
train_veg = np.hstack((train_veg,np.array(np.where(GT == 2)).T))
train_water = stack_3[GT==3]
train_water = np.hstack((train_water,np.array(np.where(GT == 3)).T))
train_road = stack_3[GT==4]
train_road = np.hstack((train_road,np.array(np.where(GT == 4)).T))
train_soil = stack_3[GT==5]
train_soil = np.hstack((train_soil,np.array(np.where(GT == 5)).T))
whole_training_data = []
whole_training_data.append(train_building)
whole_training_data.append(train_veg)
whole_training_data.append(train_water)
whole_training_data.append(train_road)
whole_training_data.append(train_soil)

shuffled = []
for i in range(len(whole_training_data)):
    shuffled.append(shuffle_2D_matrix(whole_training_data[i], axis = 1))

train_data, test_data, train_list, test_list, test3_rows_columns = train_test_split(shuffled)

mean_c1 = band_mean_calculator(train_list[0],11)
mean_c2 = band_mean_calculator(train_list[1],11)
mean_c3 = band_mean_calculator(train_list[2],11)
mean_c4 = band_mean_calculator(train_list[3],11)
mean_c5 = band_mean_calculator(train_list[4],11)
Mean3 = np.array([mean_c1,mean_c2,mean_c3,mean_c4,mean_c5])

cov_c1 = np.cov(train_list[0].reshape(11,len(train_list[0])))
cov_c2 = np.cov(train_list[1].reshape(11,len(train_list[1])))
cov_c3 = np.cov(train_list[2].reshape(11,len(train_list[2])))
cov_c4 = np.cov(train_list[3].reshape(11,len(train_list[3])))
cov_c5 = np.cov(train_list[4].reshape(11,len(train_list[4])))
Cov3 = np.array((cov_c1,cov_c2,cov_c3,cov_c4,cov_c5))

time_3_i = time.time()
G3 = np.zeros((img.shape[0],img.shape[1],5))
for n in range(5):       # number of classes
    G3[:,:,n] = Bayes_DF(stack_3,Mean3[n,:],Cov3[n,:,:])

ML3 = np.argmax(G3.reshape(img.shape[0]*img.shape[1],5),axis = 1)
time_3_f = time.time()
print('Time spent for stack_3: ', time_3_f-time_3_i)
ML3 = ML3.reshape(img.shape[0],img.shape[1])
PlotMaps(ML3,labels,cmap,patches,'Stack_3 map')

#______________________________________
# Accuracy Assessment
row_1 = ML1[test1_rows_columns[0][:, 0].astype(int), test1_rows_columns[0][:, 1].astype(int)]
row_2 = ML1[test1_rows_columns[1][:, 0].astype(int), test1_rows_columns[1][:, 1].astype(int)]
row_3 = ML1[test1_rows_columns[2][:, 0].astype(int), test1_rows_columns[2][:, 1].astype(int)]
row_4 = ML1[test1_rows_columns[3][:, 0].astype(int), test1_rows_columns[3][:, 1].astype(int)]
row_5 = ML1[test1_rows_columns[4][:, 0].astype(int), test1_rows_columns[4][:, 1].astype(int)]
Confusion_Matrix = np.array([[1632,23,0,42,715],[23,202,0,170,0],[0,0,60,0,0],[6,42,0,128,0],[318,0,0,0,492]])
OA, CA = Accuracy(Confusion_Matrix)
print('Accuracy: Class 1 >> ', OA,CA)

row_1 = ML2[test2_rows_columns[0][:, 0].astype(int), test2_rows_columns[0][:, 1].astype(int)]
row_2 = ML2[test2_rows_columns[1][:, 0].astype(int), test2_rows_columns[1][:, 1].astype(int)]
row_3 = ML2[test2_rows_columns[2][:, 0].astype(int), test2_rows_columns[2][:, 1].astype(int)]
row_4 = ML2[test2_rows_columns[3][:, 0].astype(int), test2_rows_columns[3][:, 1].astype(int)]
row_5 = ML2[test2_rows_columns[4][:, 0].astype(int), test2_rows_columns[4][:, 1].astype(int)]
Confusion_Matrix = np.array([[1074,251,0,0,1087],[7,388,0,0,0],[0,1,59,0,0],[1,0,0,175,0],[192,1,0,0,617]])
OA, CA = Accuracy(Confusion_Matrix)
print('Accuracy: Class 2 >> ', OA,CA)

row_1 = ML3[test3_rows_columns[0][:, 0].astype(int), test3_rows_columns[0][:, 1].astype(int)]
row_2 = ML3[test3_rows_columns[1][:, 0].astype(int), test3_rows_columns[1][:, 1].astype(int)]
row_3 = ML3[test3_rows_columns[2][:, 0].astype(int), test3_rows_columns[2][:, 1].astype(int)]
row_4 = ML3[test3_rows_columns[3][:, 0].astype(int), test3_rows_columns[3][:, 1].astype(int)]
row_5 = ML3[test3_rows_columns[4][:, 0].astype(int), test3_rows_columns[4][:, 1].astype(int)]
print(np.bincount(row_1))
print(np.bincount(row_2))
print(np.bincount(row_3))
print(np.bincount(row_4))
print(np.bincount(row_5))
Confusion_Matrix = np.array([[1370,45,0,66,931],[14,312,0,69,0],[0,0,60,0,0],[0,45,0,131,0],[203,0,0,0,607]])
OA, CA = Accuracy(Confusion_Matrix)
print('Accuracy: Class 3 >> ', OA,CA)

#______________________________________
# Prior Probability
P_w = [0.4,0.1,0.05,0.05,0.4]
time_p_i = time.time()
G1_P = np.zeros((img.shape[0],img.shape[1],5))
for n in range(5):       # number of classes
    G1_P[:,:,n] = Bayes_DF(stack_1,Mean1[n,:],Cov1[n,:,:],P_w[n])

ML_p = np.argmax(G1_P.reshape(img.shape[0]*img.shape[1],5),axis = 1)
time_p_f = time.time()
print('Time spent for Stack_1 with PP: ', time_p_f-time_p_i)
ML_p = ML_p.reshape(img.shape[0],img.shape[1])
PlotMaps(ML_p,labels,cmap,patches,'Stack 1 with prior probability map')

row_1 = ML_p[test1_rows_columns[0][:, 0].astype(int), test1_rows_columns[0][:, 1].astype(int)]
row_2 = ML_p[test1_rows_columns[1][:, 0].astype(int), test1_rows_columns[1][:, 1].astype(int)]
row_3 = ML_p[test1_rows_columns[2][:, 0].astype(int), test1_rows_columns[2][:, 1].astype(int)]
row_4 = ML_p[test1_rows_columns[3][:, 0].astype(int), test1_rows_columns[3][:, 1].astype(int)]
row_5 = ML_p[test1_rows_columns[4][:, 0].astype(int), test1_rows_columns[4][:, 1].astype(int)]
print(np.bincount(row_1))
print(np.bincount(row_2))
print(np.bincount(row_3))
print(np.bincount(row_4))
print(np.bincount(row_5))
Confusion_Matrix = np.array([[1662,29,0,11,710],[24,368,0,3,0],[0,1,59,0,0],[12,96,0,68,0],[305,0,0,0,505]])
OA, CA = Accuracy(Confusion_Matrix)
print('Accuracy: Stack_1 with Prior Probability >> ', OA,CA)

#______________________________________
# Minimum Distance Classifier
time_MD_i = time.time()
G1_MD = np.zeros((img.shape[0],img.shape[1],5))
for n in range(5):       # number of classes
    G1_MD[:,:,n] = Bayes_DF(stack_1,Mean1[n],np.eye(3))

ML_MD = np.argmax(G1_MD.reshape(img.shape[0]*img.shape[1],5),axis = 1)
time_MD_f = time.time()
print('Time spent for Minimum Distance Classification: ', time_MD_f-time_MD_i)
ML_MD = ML_MD.reshape(img.shape[0],img.shape[1])
PlotMaps(ML_MD,labels,cmap,patches,'Minimum Distance map')

row_1 = ML_MD[test1_rows_columns[0][:, 0].astype(int), test1_rows_columns[0][:, 1].astype(int)]
row_2 = ML_MD[test1_rows_columns[1][:, 0].astype(int), test1_rows_columns[1][:, 1].astype(int)]
row_3 = ML_MD[test1_rows_columns[2][:, 0].astype(int), test1_rows_columns[2][:, 1].astype(int)]
row_4 = ML_MD[test1_rows_columns[3][:, 0].astype(int), test1_rows_columns[3][:, 1].astype(int)]
row_5 = ML_MD[test1_rows_columns[4][:, 0].astype(int), test1_rows_columns[4][:, 1].astype(int)]
Confusion_Matrix = np.array([[1595,5,0,196,616],[0,354,2,24,15],[0,0,60,0,0],[0,20,3,152,1],[70,0,0,0,740]])
OA, CA = Accuracy(Confusion_Matrix)
print('Accuracy: Stack_1, Minimum Distance >> ', OA,CA)