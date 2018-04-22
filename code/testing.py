from PIL import Image #Image class import from pil package
import numpy as np
import matplotlib.pyplot as plt

img6 = Image.open(r"C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\600nm.png","r")
img65 = Image.open(r"C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\650nm.png","r")
img7 = Image.open(r"C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\700nm.png","r")
img75 = Image.open(r"C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\750nm.png","r")
img8 = Image.open(r"C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\800nm.png","r")
img85 = Image.open(r"C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\850nm.png","r")
img9 = Image.open(r"C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\900nm.png","r")
img95 = Image.open(r"C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\950nm.png","r")
img1 = Image.open(r"C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\1000nm.png","r")
root = np.zeros([9])


def developN(img, root, k):
    #pix_val = list(img.getdata())
    #arr1 = np.array(img.getdata())
    #print(arr.shape)
    #img2.show()
    #half_the_width = img.size[0] / 2
    #half_the_height = img.size[1] / 2
    img4 = img.crop((0, 0, 100, 100))
    """img4 = img.crop(
            (
                    half_the_width - 50,
                    half_the_height - 12,
                    half_the_width + 50,
                    half_the_height + 50
                    )
            )
            #img4.save("700_u.png")"""
    arr = np.array(img4.getdata())
    path = str(k+9)+".png"
    img4.save(path)
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
                
    value = np.average(avg_float)
    print("avg value:\n",value)
    print("r components")
    print(r,"\n")
    print("g components")
    print(g,"\n")
    print("b components")
    print(b,"\n")
    print("alp components")
    print(alph,"\n") 
    print(k,"\n")
    root[k] = value
    return root
k=0
root = developN(img6,root,k)
k=k+1
root = developN(img65,root,k)
k=k+1
root = developN(img7,root,k)
k=k+1
root = developN(img75,root,k)
k=k+1
root = developN(img8,root,k)
k=k+1
root = developN(img85,root,k)
k=k+1
root = developN(img9,root,k)
k=k+1
root = developN(img95,root,k)
k=k+1
root = developN(img1,root,k)
k=k+1
