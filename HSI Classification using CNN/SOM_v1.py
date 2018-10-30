# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:37:51 2018

@author: user
"""

#Import the library
import SimpSOM as sps
import os
from spectral import *
import scipy.io as sio

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "data/92AV3C.lan"
abs_file_path = os.path.join(script_dir, rel_path)
rel_path2 = "data/Indian_pines_gt.mat"
abs_file_path2 = os.path.join(script_dir,rel_path2)

gt = sio.loadmat(abs_file_path2)
gtd = gt['indian_pines_gt']

img = open_image(abs_file_path)
imgX = img.load()
imgX = imgX.reshape(145*145,220)
raw_data = imgX
#Build a network 20x20 with a weights format taken from the raw_data and activate Periodic Boundary Conditions. 
net = sps.somNet(20, 20, raw_data, PBC=True)

#Train the network for 10000 epochs and with initial learning rate of 0.1. 
net.train(0.01, 10)

#Save the weights to file
net.save('filename_weights')

#Print a map of the network nodes and colour them according to the first feature (column number 0) of the dataset
#and then according to the distance between each node and its neighbours.
net.nodes_graph(colnum=0)
net.diff_graph()

#Project the datapoints on the new 2D network map.
net.project(raw_data, labels=gtd.flatten())

#Cluster the datapoints according to the Quality Threshold algorithm.
net.cluster(raw_data, type='qthresh')	