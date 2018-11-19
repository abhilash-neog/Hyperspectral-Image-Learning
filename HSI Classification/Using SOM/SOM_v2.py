from minisom import MiniSom
import numpy as np
import os
import pickle
from spectral import *
import scipy.io as sio
import matplotlib.pyplot as plt
#import sys
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "data/92AV3C.lan"
abs_file_path = os.path.join(script_dir, rel_path)
rel_path2 = "data/Indian_pines_gt.mat"
abs_file_path2 = os.path.join(script_dir,rel_path2)

gt = sio.loadmat(abs_file_path2)
gtd = gt['indian_pines_gt']
gtd = gtd.flatten()
img = open_image(abs_file_path)
imgX = img.load()
imgX = imgX.reshape(145*145,220)
raw_data = imgX


map_dim = 145

alp = np.argsort(gtd)
dat = raw_data[alp]
dat = dat[10776:]

alx = np.sort(gtd)
alx = alx[10776:]

def dimensionality_reduction(dat):
    #dat = dat.reshape(145*145,220)
    dat = StandardScaler().fit_transform(dat)
    pca = PCA(n_components = 100, svd_solver='randomized',whiten=True)
    print("explained_variance_ration:",sum(pca.fit(dat).explained_variance_ratio_))
    principal_components = pca.fit_transform(dat)
    
    principal_components = principal_components.reshape(21025,100)
    
    return principal_components,pca



dat,pca = dimensionality_reduction(imgX)
map_dim1 = 145
map_dim2 = 145
som = MiniSom(map_dim1, map_dim2, 100, sigma=4.0, learning_rate=0.5,neighborhood_function='gaussian')
#som.random_weights_init(W)
som.pca_weights_init(dat)
print("Training...")
som.train_random(dat, 10000)
print("\n...ready!")

qnt = som.quantization(dat)

qnt = pca.inverse_transform(qnt)
qnt = qnt.reshape(145,145,220)

with open('som_10000.p', 'wb') as outfile:
    pickle.dump(som, outfile)

imshow(qnt,(10,20,50))
#plt.figure(figsize=(16, 16))
#wmap = {}
#im = 0
#for x, t in zip(dat, alx):  # scatterplot
#    w = som.winner(x)
#    wmap[w] = im
#    plt.text(w[0]+np.random.rand()*.9,  w[1]+np.random.rand()*.9, str(t), color=plt.cm.rainbow(t / 16.), fontdict={'weight': 'bold',  'size': 14})
#    im = im + 1
#plt.axis([0, som.get_weights().shape[0], 0,  som.get_weights().shape[1]])
#plt.savefig('som_pines.png')
#plt.show()
#qnt = som.quantization(raw_data)
#print(qnt)



