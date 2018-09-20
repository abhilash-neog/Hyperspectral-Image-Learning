from spectral import *
from keras.layers import Dense, Conv1D, Activation, MaxPooling1D, Input,Conv2D,MaxPooling2D, Flatten,Dropout
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
import scipy.io as sio
import numpy as np
from keras import optimizers
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import os
from numpy import linalg as LA
from sklearn.decomposition import PCA


script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "data/92AV3C.lan"
abs_file_path = os.path.join(script_dir, rel_path)

img = open_image(abs_file_path)

imgX = img.load()

gt = sio.loadmat(r'C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\HSI Classification using CNN\data\Indian_pines_gt.mat')
gtd = gt['indian_pines_gt']#target


#calculation of f-norm
#imgX.shape->145,145,220 - I1 -145,I2-145,I3-220
fnorms = []
for i in range(0,imgX.shape[2]):
    fnorms.append(LA.norm(imgX[:,:,i]))

#len(fnorms)
ind = np.argsort(fnorms)
#ind- 220
index = ind[:150]

a = imgX[:,:,index[0]]

for i in index[1:]:
    b = imgX[:,:,i]
    a = np.dstack((a,b))

#dimensionality reduction

#a is the new HSI
X_train = a
y_train = gtd


def labelEncode(labels):
    #one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)
    lab_encoder = LabelEncoder()
    int_encoder = lab_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    int_encoder = int_encoder.reshape(len(int_encoder), 1)
    onehot_encoded = onehot_encoder.fit_transform(int_encoder)
    return onehot_encoded

def get_model():
    inputs = Input(shape=(220,1,1))
    #model = Sequential()
    #"""
    x = Conv2D(20,kernel_size = (6,1), activation = 'tanh')(inputs)
    x = MaxPooling2D(pool_size = (3,1))(x)
    x = Dropout(0.004)(x)
    
    x = Conv2D(10,kernel_size = (6,1), activation = 'tanh')(x)
    x = MaxPooling2D(pool_size = (2,1))(x)
    #x = BatchNormalization()(x)
    x = Dropout(0.02)(x)
    
    x = Conv2D(10,kernel_size = (3,1), activation = 'tanh')(x)
    x = MaxPooling2D(pool_size = (2,1))(x)
    
    x = x = Conv2D(10,kernel_size = (3,1), activation = 'tanh')(x)
    x = MaxPooling2D(pool_size = (2,1))(x)
    x = Dropout(0.02)(x)
    
    x = Flatten()(x)
    
    x = Dense(100,activation = 'tanh')(x)
    x = Dense(50, activation = 'tanh')(x)
    x = Dense(24,activation = 'tanh')(x)
    output = Dense(17, activation = 'softmax')(x)
    model = Model(inputs=inputs, outputs=output)
    #"""
    """
    model.add(Conv1D(20,kernel_size = 25, input_dim = 220))
    #Input size should be [batch, 1d, 2d, ch] = (None, 1, 15000, 1)
    model.add(Activation('tanh'))
    model.add(MaxPooling1D(pool_size = 6))
    model.add(Dense(100))
    model.add(Activation('tanh'))
    model.add(Dense(16))
    model.add(Activation('softmax'))
    """
    return model

y_train = labelEncode(y_train)
#y_test = labelEncode(y_test)
model = get_model()

#opt = optimizers.SGD(lr=0.2, decay=1e-6, momentum=0.9, nesterov=True)
#opt = optimizers.Adam(lr=0.001, decay=1e-6)
opt = optimizers.Adagrad(lr=0.05, epsilon=None, decay=1e-4)

model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

X_train = np.array(X_train).reshape(len(X_train),len(X_train[0]),len(X_train[0][0]),1)
#X_test = np.array(X_test).reshape(len(X_test),len(X_test[0]),len(X_test[0][0]),1)
#fl = int(len(X_train)/225)
fl = 5
kf = KFold(n_splits = fl, shuffle = True, random_state = 1)
folds = list(kf.split(X_train,y_train))

for j, (train_id, val_id) in enumerate(folds):  
    print('\nFold ',j)
    X_train_kf = X_train[train_id]
    y_train_kf = y_train[train_id]
    X_valid_kf = X_train[val_id]
    y_valid_kf = y_train[val_id]
    model.fit(np.array(X_train),y_train,epochs = 20, batch_size = 8)
    score = model.evaluate(np.array(X_valid_kf),y_valid_kf, batch_size = 8)
    print(score)
    


#test accuracy - 61% -> improvement - TRUE
