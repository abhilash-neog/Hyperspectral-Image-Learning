from minisom import MiniSom
import numpy as np
import os
from spectral import *
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
#import sys
#from sklearn.preprocessing import scale
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
dat = raw_data
alx = gtd
#alp = np.argsort(gtd)
#dat = raw_data[alp]
#dat = dat[10776:]

#alx = np.sort(gtd)
#alx = alx[10776:]



def dimensionality_reduction(dat):
    #dat = dat.reshape(145*145,220)
    dat = StandardScaler().fit_transform(dat)
    pca = PCA(n_components = 100, svd_solver='randomized',whiten=True)
    print("explained_variance_ration:",sum(pca.fit(dat).explained_variance_ratio_))
    principal_components = pca.fit_transform(dat)
    
    principal_components = principal_components.reshape(145*145,100)
    
    return principal_components


dat = dimensionality_reduction(dat)
map_dim1 = 50
map_dim2 = 50#100
som = MiniSom(map_dim1, map_dim2, 100, sigma=0.8, learning_rate=0.5,neighborhood_function='gaussian')
#som.random_weights_init(W)
#som.pca_weights_init(dat)
som.random_weights_init(dat)
print("Training...")
som.train_random(dat, 10000)
print("\n...ready!")

plt.figure(figsize=(20, 20))
wmap = {}
im = 0
for x, t in zip(dat, alx):  # scatterplot
    w = som.winner(x)
    wmap[w] = im
    plt.text(w[0]+np.random.rand()*.9,  w[1]+np.random.rand()*.9, str(t/5), color=plt.cm.rainbow(t/16), fontdict={'size': 18})
    im = im + 1
plt.axis([0, som.get_weights().shape[0], 0,  som.get_weights().shape[1]])
#plt.savefig('som_pines_1.pdf')
plt.show()

qnt = som.quantization(dat)
#print(qnt)
#qnt = som.quantization_error(dat)
print(qnt)

with open('som_model_2.p','wb') as outfile:
    pickle.dump(som,outfile)

#som.quantization_error(dat)
#Out[131]: 9.607098806680474
    
#16X100
#som.quantization_error(dat)
#Out[134]: 9.703761218380942


#qe : Average distance between each data vector and its BMU.
#       Measures map resolution.
"""
max_iter = 500
q_error_pca_init = []
iter_x = []
for i in range(max_iter):
    percent = 100*(i+1)/max_iter
    rand_i = np.random.randint(len(raw_data))
    som.update(raw_data[rand_i], som.winner(raw_data[rand_i]), i, max_iter)
    if (i+1) % 100 == 0:
        error = som.quantization_error(raw_data)
        q_error_pca_init.append(error)
        iter_x.append(i)
        sys.stdout.write(f'\riteration={i:2d} status={percent:0.2f}% error={error}')

plt.plot(iter_x, q_error_pca_init)
plt.ylabel('quantization error')
plt.xlabel('iteration index')
plt.savefig("testresults.pdf")
"""

