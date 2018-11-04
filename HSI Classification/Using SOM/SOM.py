
# coding: utf-8

# 
# # Self Organizing Maps example - California Housing

# In this document, the SOMPY lib is going to be used in order to provide an example of usage of the Self Organising Maps algorithm. The data to be used will be the California Housing dataset, included in the SciKit Learn library and included below

# In[3]:

#get_ipython().magic('matplotlib inline')
import numpy as np
import os
import scipy.io as sio
from spectral import *
from matplotlib import pyplot as plt
from sompy.sompy import SOMFactory
#from sklearn.datasets import fetch_california_housing


# ## Data Loading

# First of all the data is loaded into the local environment as a numpy array. 

# In[2]:

#data = fetch_california_housing()
#descr = data.DESCR
#names = fetch_california_housing().feature_names+["HouseValue"]
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "data/92AV3C.lan"
abs_file_path = os.path.join(script_dir, rel_path)
rel_path2 = "data/Indian_pines_gt.mat"
abs_file_path2 = os.path.join(script_dir,rel_path2)

gt = sio.loadmat(abs_file_path2)

gtd = gt['indian_pines_gt']
target = gtd.flatten()
img = open_image(abs_file_path)
imgX = img.load()
imgX = imgX.reshape(145*145,220)
names = [i for i in range(0,15)]
data = np.column_stack([imgX, target])
#print(descr)
#print( "FEATURES: ", ", ".join(names))


# ## SOM Training

# The SOM training consists in 2 phases: the rough and the finetune one. The parameters that can be configured in the training step are:
# 
# - The size of each individual grid
# - The rough and finetune iterations
# - The rough and finetune initial and final radiuses
# - The initialization mechanism (random/pca)
# 
# For the current example, only the rough/finetune iterations and the initialization mechanism parameters have been chosen. The other ones have not been specified so that the algorithm will choose them authomatically.  
# 
# For quantifying the error of the approximation, 2 metrics should be computed: 
# 
# - **The quantization error**: average distance between each data vector and its BMU.
# - **The topographic error**: the proportion of all data vectors for which first and second BMUs are not adjacent units.
# 
# A rule of thumb is to generate several models with different parameters and choose the one which, having a topographic error very near to zero, has the lowest quantization error. It is important to hold the topographic error very low in order to make the components smooth and easy to understand. 

# In[3]:

#msz = calculate_msz(data)
sm = SOMFactory().build(data, normalization = 'var', initialization='random', component_names=names)
sm.train(n_job=1, verbose=False, train_rough_len=2, train_finetune_len=5)


# In[4]:

topographic_error = sm.calculate_topographic_error()
quantization_error = np.mean(sm._bmu[1])
print ("Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error))


# ## Visualization

# #### Components map

# In[5]:

from sompy.visualization.mapview import View2D
view2D  = View2D(10,10,"rand data",text_size=10)
view2D.show(sm, col_sz=4, which_dim="all", denormalize=True)


# #### Hits map

# In[6]:

from sompy.visualization.bmuhits import BmuHitsView

vhts  = BmuHitsView(10,10,"Hits Map",text_size=7)
vhts.show(sm, anotate=True, onlyzeros=False, labelsize=12, cmap="Greys", logaritmic=False)


# #### K-Means clustering

# In[7]:

from sompy.visualization.hitmap import HitMapView
sm.cluster(4)
hits  = HitMapView(10,10,"Clustering",text_size=7)
a=hits.show(sm, labelsize=12)


# ## Conclusions

# From the visualizations above we can extract different conclusions like,
# - The houses which have, on average, more bedrooms are generally on lower average income areas.
# - The highest occupations occours only in cities where the population is high.
# - The latitude and longitude of the samples have a strong negative correlation. It can be because California is diagonally oriented with respect to the coordinates system.
# - The most demanded houses (AveOccup) are placed on the 37~38 latitude and -121.6~-121 longitude; i.e. near San-Francisco area
# - Old houses are more likely to have less rooms and bedrooms on average.
# - Low average income areas usually have less rooms and bedrooms than high average ones
# - The house value seems to be related with the average income of the area where it sits.
# 
# It is important to remark that there are areas on the map where the density of instances is lower than others. It is represented by the hit map and it should be taken in consideration when interpreting the components map.
# 
# The clustering map can be used to help to find out the different behaviors represented in the components map.
