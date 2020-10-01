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

    newImage    = image.resize((newWidth, newHeight))
    return newImage

def merge_two_images(img1: Image.Image
                    ,img2: Image.Image,
                     alpha=0.25):
    image3 = changeImageSize(800, 500, img1)
    image4 = changeImageSize(800, 500, img2)

    image5 = image3.convert("RGBA")
    image6 = image4.convert("RGBA")

    alphaBlended1 = Image.blend(image5, image6, alpha=alpha)

    return alphaBlended1
def convertToPILImg(img1: np.array)-> Image.Image:
    numpy_image_rescaled = img1 * 255
    # converting the dfloat64 numpy to a unit8 - is required by PIL
    numpy_image_rescaled_uint8 = np.array(numpy_image_rescaled, np.uint8)
    # convert to PIL and show
    im = Image.fromarray(numpy_image_rescaled_uint8)
    return im