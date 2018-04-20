from PIL import Image #Image class import from pil package
import numpy as np
import matplotlib.pyplot as plt

img = Image.open(r"C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\700nm.png","r")
#img2 = img.crop((0, 0, 100, 100))
pix_val = list(img.getdata())
arr1 = np.array(img.getdata())
#print(arr.shape)
#img2.show()
half_the_width = img.size[0] / 2
half_the_height = img.size[1] / 2
img4 = img.crop(
    (
        half_the_width - 18,
        half_the_height - 36,
        half_the_width + 44,
        half_the_height + 32
    )
)
img4.save("700_u.png")
arr = np.array(img4.getdata())
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
plt.savefig(r"C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\700nm_updated.pdf")