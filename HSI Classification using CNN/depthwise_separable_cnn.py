# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:41:57 2018

@author: admin
"""

import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense,Flatten,Activation,Dropout
from keras.layers import Convolution1D,Convolution2D