import tensorflow as tf
import os,sys,inspect
from PIL import Image, ImageEnhance
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from general_image_func import changeImageSize,merge_two_images


import tensorflow as tf
from PIL import Image, ImageEnhance

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy

class brightness:
    """class for adjusting the brightness of images

    Returns:
        PIL.Image.Image: PIL.Image.Image is a image format easyly used with .show
    """
    #default value
    factor = 0.5

    def __init__(self,config:dict):
        """Instatiation of the class with a config file will overide the default values

        Args:
            config (dict): The input should be a dictionary where the key is the name of the variable to change,
            The value would then be the value refresen by the keyword.
            keys = ['factor']
        """
        keys = ['factor']

        if keys[0] in config:
            self.factor = config.get(keys[0])

    # Method to adjust the brightness, a lower factor value will result in a darker picture.
    def AdjustBrightness(self,img):
        """Adjusts the brightness of the image, using the inputtet factor.

        Args:
            img (PIL.Image.Image): PIL image file

        Returns:
            PIL.Image.Image: PIL file
        """
        IEB = ImageEnhance.Brightness(img)
        return IEB.enhance(self.factor)

    # Method to adjust the color, a lower factor value will result in dimmer colors.
    def AdjustColor(self,img):
        """Method to adjust the color, a lower factor value will result in dimmer colors.

        Args:
            img (PIL.Image.Image): PIL image

        Returns:
            PIL.Image.Image: PIL file
        """
        IEC = ImageEnhance.Color(img)
        return IEC.enhance(self.factor)

    # Creates a new picture with the color needed.
    def GetRGB(self,r:int, g:int, b:int, maxWidth:int, maxHeight:int):
        return Image.new("RGBA", (maxWidth, maxHeight), (r, g, b, 255))

    # Adjust the pictures brightness depending on day or night. Where a factor between 1 will result in night and above in day.
    def DayAdjustment(self,img):
        img = self.AdjustBrightness(img)
        if self.factor < 1:
            img = self.AdjustColor(img)
            blue = self.GetRGB(0,0,220, 800, 500)
            return merge_two_images(img, blue,alpha=0.2)
        else:
            return img
    
def QuickDebug():
    img = Image.open("C:\\Users\\jeppe\\Desktop\\GTSRB_Final_Training_Images\\GTSRB\\Final_Training\\Images\\00000\\00002_00029.ppm")
    day = {'factor':1.3} 
    night = {'factor':0.3}
    bright = brightness(day)
    dark = brightness(night)
    bright.DayAdjustment(img).show()
    dark.DayAdjustment(img).show()



