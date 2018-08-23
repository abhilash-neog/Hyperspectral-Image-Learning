from spectral import *
from keras.layers import Dense, Conv1D, Activation, MaxPooling1D, Input
from keras.models import Sequential, Model
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split

img = open_image(r'C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\Hyperspectral Image Visualization\92AV3C.lan')

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
gt = sio.loadmat(r'C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\HSI Classification using CNN\data\Indian_pines_gt.mat')
gtd = gt['indian_pines_gt']#target

imgN = np.empty([21025,220,1])#feature vectors
k = 0
print(imgN.shape)
for i in range(0,len(imgX)):
    for j in range(0,imgX[i].shape[1]):
        imgN[k] = imgX[i,j,:].reshape(220,1)#(1,220)
        k+=1
        
Y = gtd.flatten()#feature labels

X_train, X_test, y_train, y_test = train_test_split(imgN,Y, test_size = 0.45)

def get_model():
    inputs = Input(shape=(220,))
    #model = Sequential()
    x = Conv1D(20,kernel_size = 25, strides = 1, activation = 'tanh')(inputs)
    x = MaxPooling1D(pool_size = 6,strides = 1)(x)
    x = Dense(100,activation = 'tanh')(x)
    output = Dense(16, activation = 'softmax')(x)
    model = Model(inputs=inputs, outputs=output)
    #model.add(Conv1D(20,kernel_size = 25, strides = 1,input_shape = (11563,220,1)))
    #Input size should be [batch, 1d, 2d, ch] = (None, 1, 15000, 1)
    #model.add(Activation('tanh'))
    #model.add(MaxPooling1D(pool_size = 6,strides = 1))
    #model.add(Dense(100))
    #model.add(Activation('tanh'))
    #model.add(Dense(16))
    #model.add(Activation('softmax'))
    return model

model = get_model()
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train,y_train,epochs = 10,batch_size = 16)

