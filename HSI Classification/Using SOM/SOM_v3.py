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

img = plt.imread('gt.jpg')
img = img[:,:,:3]
# reshaping the pixels matrix
pixels = np.reshape(img, (img.shape[0]*img.shape[1], 3))

# SOM initialization and training
print('training...')
som = MiniSom(4, 4, 3, sigma=1.,
              learning_rate=0.2, neighborhood_function='bubble')  # 3x3 = 9 final colors
som.random_weights_init(pixels)
starting_weights = som.get_weights().copy()  # saving the starting weights
som.train_random(pixels, 500)

print('quantization...')
qnt = som.quantization(pixels)  # quantize each pixels of the image
print('building new image...')
clustered = np.zeros(img.shape)
for i, q in enumerate(qnt):  # place the quantized values into a new image
    clustered[np.unravel_index(i, dims=(img.shape[0], img.shape[1]))] = q
print('done.')

# show the result
plt.figure(figsize=(7, 7))
plt.figure(1)
plt.subplot(221)
plt.title('original')
plt.imshow(img)
plt.subplot(222)
plt.title('result')
plt.imshow(clustered)

plt.subplot(223)
plt.title('initial colors')
plt.imshow(starting_weights, interpolation='none')
plt.subplot(224)
plt.title('learned colors')
plt.imshow(som.get_weights(), interpolation='none')

plt.tight_layout()
plt.savefig('som_color_quantization.png')
plt.show()