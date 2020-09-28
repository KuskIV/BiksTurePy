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

def AddParticels(path, size=10, Opacity=120, frequency=40, LoopJump=1):
    img = Image.open(path)
    pixels = img.load()
    
    wSnow, hSnow = FlakeSize(img.size, size)

    #AddBeautifulSnowflake(img.width / 2, img.height / 2, pixels, hSnow, Opacity, img.size)

    for i in range(0, img.size[0], LoopJump):
        for j in range(0, img.size[1], LoopJump):
            if AddFlake(frequency):
                #AddSnowflake(i, j, pixels, wSnow, hSnow, Opacity, img.size)
                AddBeautifulSnowflake(i, j, pixels, hSnow, Opacity, img.size)
    
    
    NumpyToImage(np.asarray(img))

AddParticels("C:/Users/madsh/OneDrive/Code/Python/BiksTurePy/FullIJCNN2013/00/00000.ppm")
#AddParticels("C:/Users/madsh/OneDrive/Code/Python/BiksTurePy/FullIJCNN2013/00000.ppm", size=0.8, frequency=100, LoopJump=5)

