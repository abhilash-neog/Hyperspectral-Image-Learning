# Hyperspectral Image classification - identification of resources in a HSI file 

A project to classify landmarks/natural resources in a HSI file by training a deep convolutional neural network with the extracted features.

**Usage**

The code is written in python. So recommended following:
1. Python3
2. Certain libraries - Scikit, numpy, matplolib, PIL OR just download anaconda 4 :)
3. Spectral python [Spy](http://www.spectralpython.net/installation.html)
4. wxPython [Download](https://wxpython.org/)

**Current implementation**

1. The code for visualizing the HSI files that can generates the graphs for any pixel in an image and also the spectral signature of any class.

2. The HSI classifier code models a network to learn the features of every class's pixels ,to be able to classify any given pixel accurately. Level of accuracy - 61%

3. Tumor Image plotting is a simple workaround with the pillow library, extracting pixels from image channels and plotting a graph


