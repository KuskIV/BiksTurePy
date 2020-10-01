import tensorflow as tf
import os,sys,inspect
from PIL import Image, ImageEnhance
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from General_image_func import changeImageSize,merge_two_images,convertToPILImg

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

merge_two_images(img1, img2, alpha=0.3)

