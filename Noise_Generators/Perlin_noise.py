import numpy as np
import random as rd
import os,sys,inspect
import math
import matplotlib.pyplot as plt
from PIL import Image
import noise
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from data import display_numpy_image

def changeImageSize(maxWidth: int, 
                    maxHeight: int, 
                    image:np.array):
    
    widthRatio  = maxWidth/image.size[0]
    heightRatio = maxHeight/image.size[1]

    newWidth    = int(widthRatio*image.size[0])
    newHeight   = int(heightRatio*image.size[1])

    newImage    = image.resize((newWidth, newHeight))
    return newImage

def merge_two_images(img1: Image.Image, img2: Image.Image):
    image3 = changeImageSize(800, 500, img1)
    image4 = changeImageSize(800, 500, img2)

    image5 = image3.convert("RGBA")
    image6 = image4.convert("RGBA")

    alphaBlended1 = Image.blend(image5, image6, alpha=.25)

    return alphaBlended1
def convertToPILImg(img1: np.array)-> Image.Image:
    numpy_image_rescaled = img1 * 255
    # converting the dfloat64 numpy to a unit8 - is required by PIL
    numpy_image_rescaled_uint8 = np.array(numpy_image_rescaled, np.uint8)
    # convert to PIL and show
    im = Image.fromarray(numpy_image_rescaled_uint8)
    return im

def perlin_array(shape:tuple = (200, 200),
			scale:int=100, octaves:int = 6, 
			persistence:float = 0.5, 
			lacunarity:float = 2.0, 
			seed:int = None)->list:

    if not seed:

        seed = np.random.randint(0, 100)
        print("seed was {}".format(seed))

    arr = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            arr[i][j] = noise.pnoise2(i / scale,
                                        j / scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=1024,
                                        repeaty=1024,
                                        base=seed)
    max_arr = np.max(arr)
    min_arr = np.min(arr)
    norm_me = lambda x: (x-min_arr)/(max_arr - min_arr)
    norm_me = np.vectorize(norm_me)
    arr = norm_me(arr)
    return arr

def Foggyfy(img):
    perlin = perlin_array()
    return merge_two_images(convertToPILImg(perlin),img)

img2 = Image.open("C:\\Users\\jeppe\\Desktop\\GTSRB_Final_Training_Images\\GTSRB\\Final_Training\\Images\\00000\\00002_00029.ppm")
Foggyfy(img2).show()
