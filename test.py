from PIL import Image #Image class import from pil package

img=Image.open(r"C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\800nm.png","r")
pix_val = list(img.getdata())
#pix_val_flat = [x for sets in pix_val for x in sets]
print(pix_val)
r = img.split()

#creating an object of image class. now img is a pillow object
#print(img.size) #size is a prop of the img obj created
#print(img.format)
#img.show()  # terminal cannot display image.what this fn does is that it temporarily sends/display the image in the default image viewer
#the default viewer pops up on running the file

#cropping the image..make sure to comment other portions bfre running
#area=(100,100,300,375)
#cropped_img=img.crop(area)
#cropped_img.show() #img opens up in the imge viewer,the default in m laptop

# Combining images together..comment the above to run this

#within=Image.open("within.jpg")
#new=Image.open("new1.jpg")
#area=(100,100,1245,2041)
#within.paste(new,area)
#within.show()

# working on pixels..every img is made up of r g b.

#print(img.mode) # gives rgb

"""r, g, b = img.split()  #it splits into 3 parts .three var..thrfre stored in 3 variables
r.show()  #here all the three colors are not combined
g.show()
b.show()"""
#red filter pixel values. Similarly for other channels
#for every pixel take out its r, g, b values and take the average.