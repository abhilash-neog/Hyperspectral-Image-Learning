
# coding: utf-8

# In[38]:

from spectral import *


# In[21]:

img = open_image(r'C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\Hyperspectral Image Visualization\92AV3C.lan')


# In[22]:

img.__class__


# In[23]:

img


# In[24]:

img.shape


# In[25]:

type(img)


# In[26]:

img


# In[27]:

img[50,100]


# In[28]:

pixel = img[50,100]
pixel.shape


# In[29]:

# a particular band
band1 = img[:,:,0]
type(band1)


# In[30]:

band1.shape


# In[31]:

# when the spyfile object is first created only the metadata of the image is loaded. the various functions/subscript 
# operator of spyfile like shape
# etc provide information of the data loaded


# In[32]:

#loading the entire image into memory

imgA = img.load()


# In[33]:

print(imgA.shape)
imgA


# In[34]:

imgA.__class__


# In[35]:

imgA.info()


# In[17]:

#image display


# In[36]:

pylab


# In[60]:

#view1 = imshow(img)
view1 = imshow(img)


# In[40]:

#visualizing at different bands
view2 = imshow(img,(30,30,8))#Optional list of indices for bands to use in the red, green, and blue channels, respectively.


# In[41]:

view1


# In[42]:

view2


# In[43]:

viewx = imshow(img,(0,1,2))


# In[44]:

#class map display

qt = open_image(r'C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\Hyperspectral Image Visualization\92AV3GT.gis')


# In[62]:

qt.__class__


# In[63]:

qt.shape


# In[89]:

qt #has only 1 band


# In[45]:

band = qt.read_band(0)


# In[46]:

vw = imshow(classes = band)


# In[47]:

imshow(classes = band)


# In[68]:

band.shape


# In[ ]:

#uses the same arguments as imshow but with the saved image file name as the first argument.


# In[92]:

#displaying image bands alongwith color classes_overlayed on 1 band image

viewM = imshow(qt,(0,0,0),classes = band)
viewM.set_display_mode('overlay')
viewM.class_alpha = 0.7


# In[48]:

#multi banded image

viewM = imshow(img,(30,10,110),classes = band)
viewM.set_display_mode('overlay')
viewM.class_alpha = 0.2


# In[67]:

#increasing alpha for class colors

viewM = imshow(img,(30,10,110),classes = band)
viewM.set_display_mode('overlay')
viewM.class_alpha = 0.6


# In[50]:

pylab


# In[53]:

viewM = imshow(img,(30,10,110),classes = band)
viewM.set_display_mode('overlay')
viewM.class_alpha = 0.6


# In[56]:

import spectral.io.aviris as aviris

img.bands = aviris.read_aviris_bands(r'C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\Hyperspectral Image Visualization\92AV3C.spc')


# In[59]:

#view1 = imshow(img)
view1 = imshow(img)


# In[61]:

view_cube(img, bands=[29, 19, 9])


# In[96]:

data = open_image(r'C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\Hyperspectral Image Visualization\92AV3C.lan').load()

gt = open_image(r'C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\Hyperspectral Image Visualization\92AV3GT.GIS').read_band(0)

pc = principal_components(data)
#Covariance.....done

xdata = pc.transform(data)

w = view_nd(xdata[:,:,:15], classes=gt)

