import scipy.io as sio
import pandas as pd

matgt = sio.loadmat(r'C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\HIS Classification using CNN\data\Indian_pines_gt.mat')
#print(mat)
matxgt = matgt['indian_pines_gt']
#print(type(matxgt))
#print(matxgt.shape)
#145 * 145 pixels present each belonging to a particular class out of 16

df = pd.read_csv(r'C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\HIS Classification using CNN\data\indian pines.csv')

mat = sio.loadmat(r'C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\HIS Classification using CNN\data\Indian_pines_corrected.mat')
#print(mat)
matd = mat['indian_pines_corrected']
//print(matd)

import matplotlib.pyplot as plt
spec = matd[:,:,:3]
plt.imshow(spec)
plt.axis('off')