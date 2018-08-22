import scipy.io as sio
import pandas as pd

#matgt = sio.loadmat(r'C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\HIS Classification using CNN\data\Indian_pines_gt.mat')
#print(mat)
#matxgt = matgt['indian_pines_gt']
#print(type(matxgt))
#print(matxgt.shape)
#145 * 145 pixels present each belonging to a particular class out of 16

mat = sio.loadmat(r'C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\HSI Classification using CNN\data\Indian_pines_corrected.mat')
#type(mat) ->dict

#loading the ground truth/class file
gt = sio.loadmat(r'C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\HSI Classification using CNN\data\Indian_pines_gt.mat')

matd = mat['indian_pines_corrected']#pixel values

gtd = gt['indian_pines_gt']#target

