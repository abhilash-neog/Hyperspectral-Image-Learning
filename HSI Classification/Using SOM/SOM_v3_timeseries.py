# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 21:48:14 2018
Multivariate chemical image fusion of vibrational spectroscopic imaging modalities
@author: abhilash
"""

from minisom import MiniSom
import numpy as np
import os
import pickle
from spectral import *
import scipy.io as sio
import matplotlib.pyplot as plt
#import sys
#from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "data/timeseries_datasets/M7H16.mat"
abs_file_path = os.path.join(script_dir, rel_path)
rel_path2 = "data/timeseries_datasets/WL.txt"
abs_file_path2 = os.path.join(script_dir,rel_path2)

dat = sio.loadmat(abs_file_path)
dat = dat['M7H16'][0]

dat_train1 = dat[0][1]
dat_test1 = dat[15][1]

dat_train1 = dat_train1.reshape(191*151,121)
dat_test1 = dat_test1.reshape(191*151,121)

dat_test2 = dat[24][1]


#alp = np.argsort(gtd)
#dat = raw_data[alp]
#dat = dat[10776:]

#alx = np.sort(gtd)
#alx = alx[10776:]


#raw_data = imgX


map_dim = 145

#alp = np.argsort(gtd)
#dat = raw_data[alp]
#dat = dat[10776:]
#alx = np.sort(gtd)
#alx = alx[10776:]

def dimensionality_reduction(dat):
    #dat = dat.reshape(145*145,220)
    dat = StandardScaler().fit_transform(dat)
    pca = PCA(n_components = 60, svd_solver='randomized',whiten=True)
    print("explained_variance_ration:",sum(pca.fit(dat).explained_variance_ratio_))
    principal_components = pca.fit_transform(dat)
    
    principal_components = principal_components.reshape(191*151,60)
    #principal_components = principal_components.reshape(21025,100)

    
    return principal_components,pca


imgX,pca = dimensionality_reduction(dat_train1)
dat_test1,pcaTest = dimensionality_reduction(dat_test1)

map_dim1 = 145
map_dim2 = 145
som = MiniSom(map_dim1, map_dim2, 60, sigma=4.0, learning_rate=0.5,neighborhood_function='gaussian')

#som.random_weights_init(W)
som.pca_weights_init(imgX)
print("Training...")
som.train_random(imgX, 15000)
print("\n...ready!")


with open('som_model_timeseries1.p','wb') as outfile:
    pickle.dump(som,outfile)


qnt = som.quantization(dat_test1)

#qe : Average distance between each data vector and its BMU.
#Measures map resolution.

qnt = pcaTest.inverse_transform(qnt)
qnt = qnt.reshape(191,151,121)


#with open('som_timeseries.p', 'wb') as outfile:
#    pickle.dump(som, outfile)

imshow(qnt,(10,20,50))

#quantization error = 7.29
#train image -> Sample 1 0th hour
#test image -> Sampe 1 24th hour

#quantization error = 7.293588
#test image -> Sample 1 18 days

#try 18,6,9