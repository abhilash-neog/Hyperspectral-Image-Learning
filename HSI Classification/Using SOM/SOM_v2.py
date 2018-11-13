from minisom import MiniSom
import numpy as np
import os
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
def dimensionality_reduction(dat):
    #dat = dat.reshape(145*145,220)
    dat = StandardScaler().fit_transform(dat)
    pca = PCA(n_components = 50, svd_solver='randomized',whiten=True)
    print("explained_variance_ration:",sum(pca.fit(dat).explained_variance_ratio_))
    principal_components = pca.fit_transform(dat)
    
    principal_components = principal_components.reshape(145*145,50)
    
    return principal_components


raw_data = dimensionality_reduction(raw_data)
map_dim = 145
som = MiniSom(map_dim, map_dim, 50, sigma=4.0, learning_rate=0.5,neighborhood_function='gaussian')
#som.random_weights_init(W)
som.pca_weights_init(raw_data)
print("Training...")
som.train_random(raw_data, 1000)
print("\n...ready!")

plt.figure(figsize=(16, 16))
wmap = {}
im = 0
for x, t in zip(raw_data, gtd):  # scatterplot
    w = som.winner(x)
    wmap[w] = im
    plt. text(w[0]+.5,  w[1]+.5,  str(t),
              color=plt.cm.rainbow(t / 10.), fontdict={'weight': 'bold',  'size': 11})
    im = im + 1
plt.axis([0, som.get_weights().shape[0], 0,  som.get_weights().shape[1]])
#plt.savefig('som_pines.png')
plt.show()
#qnt = som.quantization(raw_data)
#print(qnt)
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

