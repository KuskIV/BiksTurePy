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
from General_image_func import changeImageSize,merge_two_images,convertToPILImg

def NumpyToImage(arr):
    img = Image.fromarray(arr)
    imgplot = plt.imshow(img)
    plt.show()

def GetPixels(path):
    return Image.open(path).load()

def Addition(p, a):
    return p + a if p + a < 255 else 255

def UpdateValue(pixel, addition):
    r, g, b = pixel

    r = Addition(r, addition)
    g = Addition(g, addition)
    b = Addition(b, addition)

    return r, g, b

def LegalSize(x, y, size):
    w, h = size
    return True if x >= 0 and x < w and y >= 0 and y < h else False

def CutEdges(x, y, xLim, yLim):
    return False if xLim <= 2 or yLim <= 2 else True if ((x == 0 or xLim - 1 == x) and (yLim - 1 == y or y == 0)) else False

def AddSnowflake(x, y, pixel, wSnow, hSnow, addition, size):
    hVal = math.floor(hSnow * 2)
    wVal = math.floor(wSnow * 2)

    for i in range(int(hVal)):
        for j in range(int(wVal)):
            if not CutEdges(i, j, hVal, wVal):
                xFlake = x - hSnow + i
                yFlake = y - wSnow + j
                if LegalSize(xFlake, yFlake, size):
                    pixel[xFlake, yFlake] = UpdateValue(pixel[xFlake, yFlake], addition)

def OneLayerAmount(layerIndex):
    return 2 * layerIndex - 1

def LeftLimit(w, layerIndex):
    return (w - OneLayerAmount(layerIndex)) / 2

def IsInsideImage(h, w, size):
    wLimit, hLimit = size
    return True if h >= 0 and w >= 0 and h < hLimit and w < wLimit else False

def CorrectShape(i, j, hSnow):
    i = i + 1
    j = j + 1

    if math.ceil(hSnow / 2) < j:
        j = hSnow - j + 1
    return True if i > LeftLimit(hSnow, j) and i <= hSnow - LeftLimit(hSnow, j) else False

def AddBeautifulSnowflake(y, x, pixel, hSnow, addition, size):
    hSnow = math.floor(hSnow) * 2
    hSnow = hSnow - 1 if hSnow % 2 == 0 else hSnow
    wSnow = 2 * (math.ceil(hSnow / 2)) - 1

    for i in range(wSnow):
        for j in range(hSnow):
            w = y + i
            h = x + j
            if IsInsideImage(h, w, size) and CorrectShape(i, j, hSnow):
                pixel[w, h] = UpdateValue(pixel[w, h], addition)

def FlakeSize(size, percent):
    width, height = size
    return math.floor((width * percent) / 100), math.floor((height * percent) / 100)

def AddFlake(Opacity):
    return True if random.randint(0, Opacity) == 0 else False

def AddParticels(img, size=4, Opacity=120, frequency=90, LoopJumpX=2, LoopJumpY=2):
    #img = Image.open(path)
    #print(type(img))
    pixels = img.load()

    wSnow, hSnow = FlakeSize(img.size, size)

    for i in range(0, img.size[0], LoopJumpX):
        for j in range(0, img.size[1], LoopJumpY):
            if AddFlake(frequency):
                AddBeautifulSnowflake(i, j, pixels, hSnow, Opacity, img.size)

    return np.asarray(img)
    #return NumpyToImage(np.asarray(img))

def generate_random_lines(imshape,slant,drop_length, raindrops):
    drops=[]    
    for i in range(raindrops):
        if slant<0:
            x= np.random.randint(slant,imshape[1])
        else:
            x= np.random.randint(0,imshape[1]-slant)
        y= np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))
    return drops
        
def add_rain(image, rain_drops=1500, drop_length=20, drop_width=2, blurr=(7,7), color=(150,150,150)):
    image.load()
    image = np.asarray(image, dtype="int32")

    imshape = image.shape    
    slant_extreme=5  
    slant= np.random.randint(-slant_extreme,slant_extreme)     
    drop_color=color   
    rain_drops= generate_random_lines(imshape,slant,drop_length, rain_drops)        
    for rain_drop in rain_drops:        
        cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)    
    image= cv2.blur(image,blurr) 

    return convertToPILImg(image, normilized=False) 

def load_image(path):
    img = Image.open(path)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def generate_and_show():
    path = "C:\\Users\\jeppe\\Desktop\\GTSRB_Final_Training_Images\\GTSRB\\Final_Training\\Images\\00000\\00002_00029.ppm"
    #path = "Images/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/00000/00002_00029.ppm"
    img = load_image(path)
    #img = add_rain(img, rain_drops=70, drop_length=7, drop_width=2, blurr=(2,2), color=(130,130,130))
    img = add_rain(img, rain_drops=40, drop_length=2, drop_width=3, blurr=(3,3), color=(200,200,200))

    plt.imshow(img)
    plt.show()

#generate_and_show()

#AddParticels("C:/Users/madsh/OneDrive/Code/Python/BiksTurePy/FullIJCNN2013/00/00000.ppm", frequency=10, LoopJumpX=3, LoopJumpY=2)
#AddParticels("C:/Users/madsh/OneDrive/Code/Python/BiksTurePy/FullIJCNN2013/00000.ppm", size=0.8, frequency=80, Opacity=50, LoopJumpX=5, LoopJumpY=5)
