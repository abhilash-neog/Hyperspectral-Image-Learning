from spectral import *
from keras.layers import Dense, Conv1D, Activation, MaxPooling1D, Input,Conv2D,MaxPooling2D, Flatten,Dropout
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
import scipy.io as sio
import numpy as np
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import os


img = open_image(r'C:\Users\admin\Hyperspectral-Image-Learning\Hyperspectral Image Visualization\92AV3C.lan')

#img = open_image(r'C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\Hyperspectral Image Visualization\92AV3C.lan')
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "data/92AV3C.lan"
abs_file_path = os.path.join(script_dir, rel_path)


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
gt = sio.loadmat(r'C:\Users\admin\Hyperspectral-Image-Learning\HSI Classification using CNN\data\Indian_pines_gt.mat')
gtd = gt['indian_pines_gt']#target

#imgN = np.empty([21025,220,1])#feature vectors
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
Y = list(Y)
X_train, X_test, y_train, y_test = train_test_split(imgN,Y, test_size = 0.60)
#X_train = list(X_train)
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
    x = Conv2D(20,kernel_size = (5,1), activation = 'tanh')(inputs)
    x = MaxPooling2D(pool_size = (3,1))(x)
    x = Dropout(0.004)(x)
    
    x = Conv2D(10,kernel_size = (5,1), activation = 'sigmoid')(x)
    x = MaxPooling2D(pool_size = (2,1))(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(10,kernel_size = (3,1), activation = 'tanh')(x)
    x = MaxPooling2D(pool_size = (2,1))(x)
    
    x = Flatten()(x)
    
    x = Dense(100,activation = 'tanh')(x)
    x = Dense(50, activation = 'tanh')(x)
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
y_test = labelEncode(y_test)
model = get_model()

sgd = optimizers.SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

X_train = np.array(X_train).reshape(len(X_train),len(X_train[0]),len(X_train[0][0]),1)
X_test = np.array(X_test).reshape(len(X_test),len(X_test[0]),len(X_test[0][0]),1)
model.fit(np.array(X_train),y_train,epochs = 20, batch_size = 32)
score = model.evaluate(np.array(X_test),y_test, batch_size = 32)

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
ada = optimizers.Adagrad(lr=0.001, decay=1e-6)
model.compile(optimizer = ada, loss = 'categorical_crossentropy', metrics = ['accuracy'])

X_train = np.array(X_train).reshape(len(X_train),len(X_train[0]),len(X_train[0][0]),1)
X_test = np.array(X_test).reshape(len(X_test),len(X_test[0]),len(X_test[0][0]),1)

model.fit(np.array(X_train),y_train,epochs = 20, batch_size = 16)
score = model.evaluate(np.array(X_test),y_test, batch_size = 16)

print(score)
#test accuracy - 58.565% -> improvement


