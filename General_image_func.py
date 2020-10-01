import numpy as np
import random as rd
from PIL import Image

def changeImageSize(maxWidth: int, 
                    maxHeight: int, 
                    image:np.array):
    
    widthRatio  = maxWidth/image.size[0]
    heightRatio = maxHeight/image.size[1]

    newWidth    = int(widthRatio*image.size[0])
    newHeight   = int(heightRatio*image.size[1])

    return image.resize((newWidth, newHeight))

def EnsureUniformImageShape(img1: Image.Image,img2: Image.Image, shape=None):
    size1 = img1.size
    size2 = img2.size

    if(shape != None):
        img1 = changeImageSize(shape[0], shape[1], img1)
        img2 = changeImageSize(shape[0], shape[1], img2)
        return img1,img2
    elif(size2 == size1):
        return img1,img2
    else:
        img2 = changeImageSize(size1[0],size1[1],img2)
        return img1,img2

def merge_two_images(img1: Image.Image
                    ,img2: Image.Image,
                     alpha:float=0.25, 
                     shape:tuple = None):
    image3,image4 = EnsureUniformImageShape(img1, img2,shape=shape)

    image5 = image3.convert("RGBA")
    image6 = image4.convert("RGBA")

    return Image.blend(image5, image6, alpha=alpha)

def convertToPILImg(img1: np.array, normilized = True)-> Image.Image:

    if normilized is True:
        numpy_image_rescaled = img1 * 255
    # converting the dfloat64 numpy to a unit8 - is required by PIL
    numpy_image_rescaled_uint8 = np.array(numpy_image_rescaled, np.uint8)
    # convert to PIL and show
    im = Image.fromarray(numpy_image_rescaled_uint8)
    return im

