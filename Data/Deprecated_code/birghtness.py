import tensorflow as tf
import os,sys,inspect
from PIL import Image, ImageEnhance
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from General_image_func import changeImageSize,merge_two_images


import tensorflow as tf
from PIL import Image, ImageEnhance

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy
 
# Method to adjust the brightness, a lower factor value will result in a darker picture.
def AdjustBrightness(img, factor):
    IEB = ImageEnhance.Brightness(img)
    return IEB.enhance(factor)

# Method to adjust the color, a lower factor value will result in dimmer colors.
def AdjustColor(img, factor):
    IEC = ImageEnhance.Color(img)
    return IEC.enhance(factor)

# Creates a new picture with the color needed.
def GetRGB(r, g, b, maxWidth, maxHeight):
    return Image.new("RGBA", (maxWidth, maxHeight), (r, g, b, 255))

# Adjust the pictures brightness depending on day or night. Where a factor between 1 will result in night and above in day.
def DayAdjustment(img, factor):
    img = AdjustBrightness(img, factor)
    if factor < 1:
        img = AdjustColor(img, factor)
        blue = GetRGB(0,0,220, 800, 500)
        return merge_two_images(img, blue,alpha=0.2)
    else:
        return img



