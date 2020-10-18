import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import  random
import  math
import os,sys,inspect
from cv2 import cv2
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from general_image_func import changeImageSize,merge_two_images,convertToPILImg

class weather:
    """The weather class is a class used to draw snow and rain onto exsistig pictures

    Returns:
        Image.Image: The image returned will have add some noise based on the config file given to the function at instatiaction.
    """
    #default values
    rain_drops = 200
    drop_length = 2
    drop_width = 2
    blurr = (5,5)
    color = (200,200,200)

    def __init__(self,config:dict):
        """Instatiation of the class with a config file will overide the default values

        Args:
            config (dict): The input should be a dictionary where the key is the name of the variable to change,
            The value would then be the value refresen by the keyword.
            keys = ['rain_drops','drop_length','drop_width,'blurr,'color']
        """
        keys = ['rain_drops','drop_length','drop_width','blurr','color']
        if keys[0] in config:
            self.rain_drops = config.get(keys[0])
        if keys[1] in config:
            self.drop_length = config.get(keys[1])
        if keys[2] in config:
            self.drop_width = config.get(keys[2])
        if keys[3] in config:
            self.blurr = config.get(keys[3])
        if keys[4] in config:
            self.color = config.get(keys[4])


    def generate_random_lines(self,imshape:tuple,slant:int)->list:
        """Function for generating random lines, it also makes the lines slant in diffrent orientations.

        Args:
            imshape (tuple(int,int)): A tuple containing the demensions of the image
            slant (int): Determins how slantet the lines should be

        Returns:
            List: List containnig the coordinats for each of the drops or flakes generated.
        """
        drops=[]    
        for i in range(self.rain_drops):
            if slant<0:
                x= np.random.randint(slant,imshape[1])
            else:
                x= np.random.randint(0,imshape[1]-slant)
            y= np.random.randint(0,imshape[0]-self.drop_length)
            drops.append((x,y))
        return drops
            
    def add_rain(self,image:Image.Image):
        """The function takes a image and adds rain or snow.

        Args:
            image (PIL.Image.Image): The input is a open PIL image

        Returns:
            PIL.Image.Image: The output is a open PIL image
        """
        image.load()
        image = np.asarray(image, dtype="int32")

        imshape = image.shape    
        slant_extreme=5  
        slant= np.random.randint(-slant_extreme,slant_extreme)     
        drop_color=self.color   
        self.rain_drops= self.generate_random_lines(imshape,slant)
        rand = random.uniform(0.5, 1)
        for rain_drop in self.rain_drops:        
            cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+int(round(self.drop_length*rand))),drop_color,int(round(self.drop_width*rand)))    
        image= cv2.blur(image,self.blurr) 

        return convertToPILImg(image, normilized=False) 

def QuickDebug():
    """Small function to test the weather class
    """
    img = Image.open("C:\\Users\\jeppe\\Desktop\\GTSRB_Final_Training_Images\\GTSRB\\Final_Training\\Images\\00000\\00002_00029.ppm")
    p = {'rain_drops':300, 'drop_length':7,'drop_width':2,'blurr':(2,2),'color':(130,130,130)}
    pn = weather(p)
    img = pn.add_rain(img).show()

#AddParticels("C:/Users/madsh/OneDrive/Code/Python/BiksTurePy/FullIJCNN2013/00/00000.ppm", frequency=10, LoopJumpX=3, LoopJumpY=2)
#AddParticels("C:/Users/madsh/OneDrive/Code/Python/BiksTurePy/FullIJCNN2013/00000.ppm", size=0.8, frequency=80, Opacity=50, LoopJumpX=5, LoopJumpY=5)
