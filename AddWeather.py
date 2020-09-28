import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import  random
import  math

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

def FlakeSize(size, percent):
    width, height = size
    return math.floor((width * percent) / 100), math.floor((height * percent) / 100) 

def AddFlake(Opacity):
    return True if random.randint(0, Opacity) == 0 else False

def AddParticels(path, size=5, Opacity=120, frequency = 40):
    img = Image.open(path)
    pixels = img.load()
    
    wSnow, hSnow = FlakeSize(img.size, size)

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if AddFlake(frequency):
                AddSnowflake(i, j, pixels, wSnow, hSnow, Opacity, img.size)
    
    NumpyToImage(np.asarray(img))

AddParticels("C:/Users/madsh/OneDrive/Code/Python/BiksTurePy/FullIJCNN2013/00/00000.ppm")
#AddParticels("C:/Users/madsh/OneDrive/Code/Python/BiksTurePy/FullIJCNN2013/00000.ppm", size=0.5, frequency=900)

