import numpy as np
import random as rd
import math
import matplotlib.pyplot as plt
from PIL import Image
import noise

def display_numpy_image(numpy_image:np.array)->None:
    """Input a (0 to 1) normalized numpy representation of a PIL image, to show it"""
    # Scaling the pixels back
    numpy_image_rescaled = numpy_image * 255
    # converting the dfloat64 numpy to a unit8 - is required by PIL
    numpy_image_rescaled_uint8 = np.array(numpy_image_rescaled, np.uint8)
    # convert to PIL and show
    im = Image.fromarray(numpy_image_rescaled_uint8)
    im.show()

"""
def psudoRand():
    seed = 4294967296
    A=1664525
    C=1
    Z = np.floor(rd.uniform(0,1)*seed)
    Z=(A*Z+C)%seed
    return Z/seed

def interpolate(a,b,x):
    ft = x* math.pi
    f = (1-math.cos(ft))* 0.5
    return a*(1-x)+b*f

def perlin1d(amp:int, wl:int, Wlimit:int):
    x = 0
    fq = 1/wl
    a = psudoRand()
    b = psudoRand()
    pos = []
    for i in range(x,Wlimit):
        if(i%wl == 0):
            a = b
            b = psudoRand()
            pos.append(a*amp)
        else:
            pos.append(interpolate(a,b,(i%wl)/wl)*amp)
    return pos

def GenerateNoise(amp,wl,oct,div,width):
    results=[]
    for i in range(oct):
        results.append(perlin1d(amp, wl,width))
        amp = amp/div
        wl = wl/div
    return results

def CombineNoise(pl):
    results = []
    for i in range(len(pl[0])):
        total = 0
        for j in range(len(pl)):
            total = total +pl[j][i]
        results.append(total)
    return results
"""
def changeImageSize(maxWidth, 
                    maxHeight, 
                    image):
    
    widthRatio  = maxWidth/image.size[0]
    heightRatio = maxHeight/image.size[1]

    newWidth    = int(widthRatio*image.size[0])
    newHeight   = int(heightRatio*image.size[1])

    newImage    = image.resize((newWidth, newHeight))
    return newImage

def merge_two_images(img1, img2):
    # Take two images for blending them together   
    # Make the images of uniform size
    image3 = changeImageSize(800, 500, img1)
    image4 = changeImageSize(800, 500, img2)

    # Make sure images got an alpha channel
    image5 = image3.convert("RGBA")
    image6 = image4.convert("RGBA")

    # alpha-blend the images with varying values of alpha
    alphaBlended1 = Image.blend(image5, image6, alpha=.25)
    #alphaBlended2 = Image.blend(image5, image6, alpha=.4)

    # Display the alpha-blended images
    return alphaBlended1
def convertToImg(img1):
    numpy_image_rescaled = img1 * 255
    # converting the dfloat64 numpy to a unit8 - is required by PIL
    numpy_image_rescaled_uint8 = np.array(numpy_image_rescaled, np.uint8)
    # convert to PIL and show
    im = Image.fromarray(numpy_image_rescaled_uint8)
    return im

def perlin2d(shape,scale,octaves,persistence,lacunarity):
    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise.pnoise2(i/scale, 
                                        j/scale, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity, 
                                        repeatx=1024, 
                                        repeaty=1024, 
                                        base=0)
    return world
def perlin_array(shape = (144, 148),
			scale=100, octaves = 6, 
			persistence = 0.5, 
			lacunarity = 2.0, 
			seed = None):

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

"""def Blend_Multiple_noise_maps(x):
    noiseMaps = []
    noiseSum = None
    for i in range(x):
        if( i== 0):
            noiseSum = convertYoImg(perlin2d((1000,1000),2.0*rd.uniform(0,1),int(round(rd.uniform(1,10))),2,0.5))
        else:
            noiseMaps.append(perlin2d((1000,1000),2.0*rd.uniform(0,1),int(round(rd.uniform(1,10))),2,0.5))
    for noise in noiseMaps:
        noiseSum = merge_two_images(noiseSum,convertYoImg(noise))
    return noiseSum
"""
"""
noiseImg1 = perlin2d((1024,1024),20.0,5,0.4,0.5)
noiseImg2 = perlin2d((1024,1024),21.0,5,0.5,0.7)
noiseImg3 = perlin2d((1024,1024),22.0,5,0.3,0.5)
noiseImg4 = perlin2d((1024,1024),23.0,5,0.6,0.6)
noiseImg5 = perlin2d((1024,1024),22.0,5,0.2,0.5)
noiseImg6 = perlin2d((1024,1024),20.0,5,0.6,0.3)
Mimg1 = merge_two_images(convertYoImg(noiseImg1),convertYoImg (noiseImg2))
Mimg2 = merge_two_images(Mimg1,convertYoImg(noiseImg3))
Mimg3 = merge_two_images(convertYoImg(noiseImg4),convertYoImg(noiseImg5))
Mimg4 = merge_two_images(Mimg3,convertYoImg(noiseImg6))

img1 = merge_two_images(Mimg2,Mimg4)
"""
def Foggyfy(img):
    perlin = perlin_array()
    return merge_two_images(convertToImg(perlin),img)
#img1 = Blend_Multiple_noise_maps(1)
#img1 = perlin_array()
#img1 = Image.open(im)
#img2 = Image.open("C:\\Users\\jeppe\\Desktop\\GTSRB_Final_Training_Images\\GTSRB\\Final_Training\\Images\\00000\\00002_00029.ppm")
#Foggyfy(img2).show()
#ims = merge_two_images(convertYoImg(img1),img2)
#ims = Image.open(ims)
#ims.show()
#display_numpy_image(img1)
#changeImageSize(90,90,ims).show()

#print(GenerateNoise())
"""
t= CombineNoise(GenerateNoise(128,128,20,4,500))
plt.plot(t)
plt.ylabel('some numbers')
plt.show()
"""
"""
def valueList():
    p = [ 151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,
         23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,
         174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,
         133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,
         89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,
         202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,
         248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,
         178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,
         14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,
         93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180]
    for i in range(256):
        p[256+i] = p[i]
    return p
            
    

def perlin2d():
    pass
def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)
def lerp(t,a,b):
    return a + t * (b - a)
def grad(hash,x,y,z):
        h = hash & 15
        u = x if h < 8 else y
        v = y if h < 4 else x if h == 12 or h==14 else z
        return (u if h&(1)==0 else -u)+( v if (h&2)==0 else -v)
def noise(x,y,z):
    p=valueList()
    floorX = math.floor(x)
    floorY = math.floor(y)
    floorZ = math.floor(z)
    X = floorX & 255
    Y = floorY & 255
    Z = floorZ & 255
    x =x - floorX
    y =y - floorY
    z =z - floorZ
    xMinus1 = x -1
    yMinus1 = y -1
    zMinus1 = z -1
    u = fade(x)
    v = fade(y) 
    w = fade(z)
    A = p[X] + Y
    AA = p[A] + Z
    AB = p[A + 1] + Z
    B = p[X + 1] + Y
    BA = p[B] + Z
    BB = p[B + 1] + Z
    return lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z), 
                            grad(p[BA], xMinus1, y, z)),
                        lerp(u, grad(p[AB], x, yMinus1, z),
                            grad(p[BB], xMinus1, yMinus1, z))),
                    lerp(v, lerp(u, grad(p[AA + 1], x, y, zMinus1),
                            grad(p[BA + 1], xMinus1, y, z - 1)),
                        lerp(u, grad(p[AB + 1], x, yMinus1, zMinus1),
                            grad(p[BB + 1], xMinus1, yMinus1, zMinus1))))
"""

"""
def Calc_Vectors_to_edges(point:list)->list:
    edges = [[0,0],[1,0],[0,1],[1,1]]
    vectors = []
    for edge in edges:
        vectors.append(edge[0]-point[0],edge[1]-point[1])
    return vectors

def Find_point(x:float, y:float)->list:
    point = [x%1.0,y%1.0]
    return point

def perlin2d(x:float, y:float)->float:
    point = Find_point(x,y)
    vectors = Calc_Vectors_to_edges(point)
"""   


