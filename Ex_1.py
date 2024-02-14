"""
Created on Wed Oct 25 09:54:31 2023

@author: Hassan Rezvan

The main file to run
"""
# Library imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from skimage.io import imread
import pandas as pd
# _____________________________________________
# Function imports
from functions import normalizing
from functions import train_test_split
from functions import shuffle_2D_matrix
from functions import band_mean_calculator
from functions import dual_band_plot
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
plt.figure(figsize=(15,20),dpi=600)
plt.imshow(CIg,cmap='PiYG')
plt.colorbar(fraction=0.047*CIg.shape[0]/CIg.shape[1])
plt.xticks([]),plt.yticks([])
plt.title('CIg map', fontsize=34)

PISI = 0.8192*blue_normalized-0.5735*nir_normalized+0.0750
# PISI = 0.8192*b3-0.5735*b4+0.0750
plt.figure(figsize=(15,20),dpi=600)
plt.imshow(PISI,cmap='gray')
plt.colorbar(fraction=0.047*PISI.shape[0]/PISI.shape[1])
plt.xticks([]),plt.yticks([])
plt.title('PISI map', fontsize=34)

EVI = 2.5*((nir_normalized-red_normalized)/(nir_normalized+6*red_normalized-7.5*blue_normalized+1))
plt.figure(figsize=(15,20),dpi=600)
plt.imshow(EVI,cmap='PiYG')
plt.colorbar(fraction=0.047*EVI.shape[0]/EVI.shape[1])
plt.xticks([]),plt.yticks([])
plt.title('EVI map', fontsize=34)

#_________________________________________________________
## Stacking
stack_1 = np.stack((b3,b2,b1),axis=2)
stack_2 = np.stack((b3,b2,b1,CIg,PISI,EVI),axis=2)
stack_3 = np.stack((b3,b2,b1,CIg,PISI,EVI,texture_mean,texture_variance,texture_homogeneity,texture_contrast,texture_disimilarity),axis=2)

#_________________________________________________________
# Ground truth data
GT = imread('GT.tif')
colors = ['white','red','green','blue','black','brown']
cmap = ListedColormap(colors)
legend_lebels = {'white':'background','red':'Building','blue':'Water',
                 'black':'Road','green':'Vegetation','brown':'Bare soil'}
patches = [Patch(color=color,label=label) for color, label in legend_lebels.items()]
plt.figure(figsize=(15,30),dpi=600)
plt.imshow(GT,cmap=cmap)
plt.title('Ground truth image', fontsize=34)
plt.xticks([]),plt.yticks([])
plt.legend(handles=patches,bbox_to_anchor=(1.22,1),facecolor='white',
           prop={'size':20})

#______________________________________
# Train-Test split
train_building = img[GT==1]
train_veg = img[GT==2]
train_water = img[GT==3]
train_road = img[GT==4]
train_soil = img[GT==5]
whole_training_data = []
whole_training_data.append(train_building)
whole_training_data.append(train_veg)
whole_training_data.append(train_water)
whole_training_data.append(train_road)
whole_training_data.append(train_soil)

shuffled = []
for i in range(len(whole_training_data)):
    shuffled.append(shuffle_2D_matrix(whole_training_data[i], axis = 1))

train_data, test_data, train_list, test_list = train_test_split(shuffled)

#______________________________________
# True color and False display
plt.figure(figsize=(15,20),dpi=600)
plt.imshow(normalizing(img[:,:,[0,1,2]]),cmap='gray')   #R-G-B
colors = [(1, 1, 1, 0),'red','green','blue','black','brown']
cmap = ListedColormap(colors)
plt.imshow(GT,cmap=cmap)
plt.xticks([]),plt.yticks([])
plt.title('True color image', fontsize=34)
plt.legend(handles=patches,bbox_to_anchor=(1.22,1),facecolor='white',
           prop={'size':20})

plt.figure(figsize=(15,20),dpi=600)
plt.imshow(normalizing(img[:,:,[3,0,1]]),cmap='gray')   #NIR-R-G
colors = [(1, 1, 1, 0),'red','green','blue','black','brown']
cmap = ListedColormap(colors)
plt.imshow(GT,cmap=cmap)
plt.xticks([]),plt.yticks([])
plt.title('False color image', fontsize=34)
plt.legend(handles=patches,bbox_to_anchor=(1.22,1),facecolor='white',
           prop={'size':20})


#______________________________________
# Histograms and bands correlation
img_reshaped = img.reshape(img.shape[0]*img.shape[1],img.shape[2])
df = pd.DataFrame(img_reshaped)
df.sample(n=5)
df.hist(figsize=(16,20), bins=50)
correlation_mat = df.corr()
print(correlation_mat)

dual_band_plot(img_reshaped)        
        
#______________________________________
# Band mean
mean_c2 = band_mean_calculator(train_veg)
mean_c5 = band_mean_calculator(train_soil)


#______________________________________
# Spectral diagram
y = [1,2,3,4]

train_veg_for_sd = train_list[1]
plt.figure(figsize=(15,15))
for i in range(len(train_veg_for_sd)):
    plt.plot(y,train_veg_for_sd[i,:],'green')
plt.plot(y,mean_c2,'black')
plt.title('Spectral diagram for Vegetation', fontsize=34) 
plt.xlabel('Band',fontsize=26),plt.ylabel('Reflectance',fontsize=26)

train_soil_for_sd = train_list[4]
plt.figure(figsize=(15,15))
for i in range(len(train_soil_for_sd)):
    plt.plot(y,train_soil_for_sd[i,:],'brown')
plt.plot(y,mean_c5,'blue')
plt.title('Spectral diagram for Soil', fontsize=34) 
plt.xlabel('Band',fontsize=26),plt.ylabel('Reflectance',fontsize=26)


