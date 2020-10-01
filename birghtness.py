import tensorflow as tf
import cv2
from PIL import Image, ImageEnhance

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy

test_pic_path = "testpic/test1.jpg"

img1 = Image.open(test_pic_path)

enhancer = ImageEnhance.Brightness(img1)

factor = 0.5
img_output = enhancer.enhance(factor)
img_output.save("testpic/darker.png")

#factor = 1.5
#im_output = enhancer.enhance(factor)
#im_output.save("testpic/brightboi.png")

img1 = Image.open("testpic/darker.png")
converter = ImageEnhance.Color(img1)
img1 = converter.enhance(0.5)
img1.save("testpic/saturatedboi.png")

img1 = Image.open("testpic/saturatedboi.png")
img2 = Image.open("testpic/blue.png")

def changeImageSize(maxWidth, 
                    maxHeight, 
                    image):
    
    widthRatio  = maxWidth/image.size[0]
    heightRatio = maxHeight/image.size[1]

    newWidth    = int(widthRatio*image.size[0])
    newHeight   = int(heightRatio*image.size[1])

    newImage    = image.resize((newWidth, newHeight))
    return newImage

def merge_two_images(img1, img2):
    # Take two images for blending them together   
    

    # Make the images of uniform size
    img3 = changeImageSize(800, 500, img1)
    img4 = changeImageSize(800, 500, img2)

    # Make sure images got an alpha channel
    img5 = img3.convert("RGBA")
    img6 = img4.convert("RGBA")

    # alpha-blend the images with varying values of alpha
    alphaBlended1 = Image.blend(img5, img6, alpha=.3)
  

   
    alphaBlended1.save("testpic/blended2.png")
merge_two_images(img1, img2)

