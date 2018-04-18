from PIL import Image #Image class import from pil package
import numpy as np
import matplotlib.pyplot as plt

img=Image.open(r"C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\800nm.png","r")
pix_val = list(img.getdata())
arr = np.array(img.getdata())
print(arr.shape)
#print(arr)
# extracting r component of each pixel
r = np.zeros([len(arr)])
g = np.zeros([len(arr)])
b = np.zeros([len(arr)])
avg = np.zeros([len(arr)])
avg_float = avg.astype(np.float)
alph = np.zeros([len(arr)])
for i in range(0,len(arr)):
    r[i] = arr[i][0]
    g[i] = arr[i][1]
    b[i] = arr[i][2]
    avg_float[i] = (r[i]+b[i]+g[i])/3
    alph[i] = arr[i][3]
     
print("r components")
print(r,"\n")
print("g components")
print(g,"\n")
print("b components")
print(b,"\n")
print("alp components")
print(alph,"\n")
#print("average of r components:\n",np.average(r))
#print("average of g components:\n",np.average(g))
#print("average of b components:\n",np.average(b))
#print("average of alpha components:\n",np.average(alph))
#r = img.split()
#for i in range(0,len(avg_float)):
  #  print(avg_float[i])
#print(avg_float)
plt.hist(avg_float, 50, normed=1, facecolor='r', alpha=0.75)
#plt.hist(g, 50, normed=1, facecolor='g', alpha=0.75)
#plt.hist(b, 50, normed=1, facecolor='b', alpha=0.75)
plt.show
plt.xlabel("Pixels ->")
plt.title("Histogram of pixel values")
plt.ylabel("Reflectance")
plt.grid(True)
plt.savefig(r"C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\histogram_plot.pdf")