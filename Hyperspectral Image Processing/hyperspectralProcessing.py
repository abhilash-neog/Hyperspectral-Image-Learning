# -*- coding: utf-8 -*-
"""
A direct executable code
Created on Wed Aug 15 20:40:38 2018
@author: abhilash

"""
from spectral import *

img = open_image('92AV3C.lan')
view = imshow(img,(100,100,100))
gt = open_image('92AV3GT.GIS').read_band(0)
view = imshow(classes=gt)

view = imshow(img,(100,100,100),classes=gt)
view.set_display_mode('overlay')
view.class_alpha = 0.3
save_rgb('gt.jpg', gt, colors=spy_colors)
