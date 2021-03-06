import numpy as np
import random as rd

import math
import matplotlib.pyplot as plt
from PIL import Image
import noise

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from general_image_func import changeImageSize,merge_two_images,convertToPILImg,display_numpy_image
from global_paths import get_paths


class perlin:
    """A class to generate psudorandom fog onto image given som paramters

    Returns:
        Image.Image: Image with a fog overlay
    """
    #Default values
    shape = (200, 200)
    scale = 100
    octaves = 6 
    persistence = 0.5
    lacunarity = 2.0
    seed = None
    alpha = 0.3
    darkness = 0.0

    def __init__(self, config:dict)->object:
        """The to configure the default variables in perlin noise.

        Args:
            config (dict): The input should be a dictionary where the key is the name of the variable to change,
            The value would then be the value refresen by the keyword.
            keys = ["shape","scale","octaves","persistence","lacunarity","seed","alpha"]
        """
        self.Keys = ["shape","scale","octaves","persistence","lacunarity","seed","alpha","darkness"]
        for key in self.Keys:
            if key in config:
                setattr(self, key, config.get(key))

    def perlin_array(self)->list:
        """
            The perlin_array method generates the actual perlin noise map.
            The noise map itself is array which will have to be converted to a PIL image later.

        Returns:
            list: The noise map is returend as a list represnting the values,
            this can be converted easyly to an image.
        """
        if not self.seed:

            seed = np.random.randint(0, 100)

        arr = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                arr[i][j] = noise.pnoise2(i / self.scale,
                                            j / self.scale,
                                            octaves=self.octaves,
                                            persistence=self.persistence,
                                            lacunarity=self.lacunarity,
                                            repeatx=1024,
                                            repeaty=1024,
                                            base=seed)
        max_arr = np.max(arr)
        min_arr = np.min(arr)
        norm_me = lambda x: (x-min_arr)/(max_arr - min_arr)
        norm_me = np.vectorize(norm_me)
        arr = norm_me(arr)
        return arr

    def Transparentfy(self,img):
        #img = Image.open('open_science_logo.png')
        img = img.convert("RGBA")
        datas = img.getdata()
        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                if item[0] > 150:
                    newData.append((0, 0, 0, 255))
                else:
                    newData.append(item)
                    
        img.putdata(newData)
        img.show()
        return img

    def get_white(self)->Image.Image:
        return Image.new("RGBA", (self.shape[0], self.shape[1]), (255, 255, 255, 255))

    def Foggyfy(self,img:Image.Image)->Image.Image:
        """
        Foggyfy is the method that applys some generated perlin noise to an image

        Args:
            img (PIL.Image.Image): The input should be the PIL image the filter should be used on.

        Returns:
            PIL.Image.Image: The returned image will of the same type and shape, but will have a perlin noise overlay.
        """
        perlin = self.perlin_array()
        
        #perlin = self.Transparentfy(convertToPILImg(perlin))
        perlin = merge_two_images(convertToPILImg(perlin), self.get_white(),alpha = self.darkness)
        return merge_two_images(perlin,img, alpha=self.alpha).convert('RGB')

    def __add__(self, img):
        return self.Foggyfy(img)

def QuickDebug()->None:
    """Small function that shows how to call the perlin class with some config dict. And shows the resulting image
    """
    img = Image.open(get_paths("dataset"))
    p = {'octaves':6, 'persistence':0.5, 'lacunarity': 2.0, 'alpha': 0.3}
    pn = perlin(p)
    img = pn.Foggyfy(img).show()

