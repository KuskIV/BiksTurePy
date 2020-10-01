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
from General_image_func import changeImageSize,merge_two_images,convertToPILImg

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
    perlin = perlin_array(shape=img.size)
    return merge_two_images(convertToPILImg(perlin),img, alpha=0.3)

#img2 = Image.open("C:\\Users\\jeppe\\Desktop\\GTSRB_Final_Training_Images\\GTSRB\\Final_Training\\Images\\00000\\00002_00029.ppm")
#Foggyfy(img2).show()
