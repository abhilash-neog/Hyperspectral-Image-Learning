from spectral import *
from keras.layers import Dense, Conv1D, Activation, MaxPooling1D, Input,Conv2D,MaxPooling2D, Flatten,Dropout
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
import scipy.io as sio
import numpy as np

import os

from keras import optimizers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "data/92AV3C.lan"
rel_path2 = "data/Indian_pines_gt.mat"
abs_file_path = os.path.join(script_dir, rel_path)
abs_file_path2 = os.path.join(script_dir,rel_path2)
img = open_image(abs_file_path)
#img.shape
#Out[2]: (145, 145, 220)

#img
#Out[3]: 
#	Data Source:   'C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\Hyperspectral Image Visualization\92AV3C.lan'
#	# Rows:            145
#	# Samples:         145
#	# Bands:           220
#	Interleave:        BIL
#	Quantization:  16 bits
#	Data format:     int16
    
#imshow(img)
#Out[4]: 
#ImageView object:
#  Display bands       :  [0, 110, 219]
#  Interpolation       :  <default>
#  RGB data limits     :
#    R: [2632.0, 4536.0]
#    G: [1017.0, 1159.0]
#    B: [980.0, 1034.0]

#loading the complete image

imgX = img.load()
#print(imgX)

#imgX.shape
#Out[10]: (145, 145, 220)

#the target features
gt = sio.loadmat(abs_file_path2)
gtd = gt['indian_pines_gt']#target

#imgN = np.empty([21025,220,1])#feature vectors
#the target features

k = 0
imgN = []
#print(imgN.shape)
for i in range(0,len(imgX)):
    for j in range(0,imgX[i].shape[1]):
        x1 = imgX[i,j,:].reshape(220,1)
        imgN.append(x1)
        #imgN[k] = imgX[i,j,:].reshape(220,1,1)#(1,220)
        k+=1
        

Y = gtd.flatten()#feature labels

#Y = gtd.flatten()#feature labels

Y = list(Y)
#X_train, X_test, y_train, y_test = train_test_split(imgN,Y, test_size = 0.20)
#X_train = list(X_train)
X_train = list(imgN)
y_train = Y

def labelEncode(labels):
    #one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
    lab_encoder = LabelEncoder()
    int_encoder = lab_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    int_encoder = int_encoder.reshape(len(int_encoder), 1)
    onehot_encoded = onehot_encoder.fit_transform(int_encoder)
    return onehot_encoded

def get_model():
    model = Sequential()
    model.add(Dense(100,input_shape=(220,1)))
    model.add(Activation('sigmoid'))
    model.add(Dense(50))
    model.add(Activation('tanh'))
    model.add(Dense(24))
    model.add(Activation('sigmoid'))
    model.add(Dense(17))
    model.add(Activation('softmax'))
    
    return model

fl = 50
kf = KFold(n_splits = fl, shuffle = True, random_state = 1)
folds = list(kf.split(X_train,y_train))
model = get_model()

for j, (train_id, val_id) in enumerate(folds):  
    print('\nFold ',j)
    X_train_kf = X_train[train_id]
    y_train_kf = y_train[train_id]
    X_valid_kf = X_train[val_id]
    y_valid_kf = y_train[val_id]
    
    model.fit(np.array(X_train),y_train,epochs = 20, batch_size = 8)
    score = model.evaluate(np.array(X_valid_kf),y_valid_kf, batch_size = 8)
    print(score)



#test accuracy - 58.565% -> improvement
    

#test accuracy - 61% -> improvement - TRUE

