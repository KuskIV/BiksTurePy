import imageio
from PIL import Image
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import general_image_func as gif 

class weather:

    #default values
    density = (0.03,0.14)
    density_uniformity = (0.8,1.0)
    drop_size = (0.3,0.4)
    drop_size_uniformity = (0.1,0.5)
    angle = (-15,15)
    speed=(0.1,0.2)
    blur=(0.001, 0.001)

    def __init__(self, config:dict):

        Keys = ['density','density_uniformity','drop_size','drop_size_uniformity','angle','speed','blur']
        if Keys[0] in config:
            self.density = config.get(Keys[0])
        if config.get(Keys[1]) != None:
            self.density_uniformity = config.get(Keys[1])
        if config.get(Keys[2]) != None:
            self.drop_size = config.get(Keys[2])
        if config.get(Keys[3]) != None:
            self.drop_size_uniformity = config.get(Keys[3])
        if config.get(Keys[4]) != None:
            self.angle = config.get(Keys[4])
        if config.get(Keys[5]) != None:
            self.speed = config.get(Keys[5])
        if config.get(Keys[6]) != None:
            self.blur = config.get(Keys[6])



def rain(image:Image.Image, density:tuple=(0.03,0.14), density_uniformity:tuple=(0.8,1.0), drop_size:tuple=(0.3,0.4), drop_size_uniformity:tuple=(0.1,0.5),angle:tuple=(-15,15), speed:tuple=(0.1,0.2), blur:tuple=(0.001, 0.001))->Image.Image: 
    """
    Method to add rain particles to images. 
    The argurments are all tupples of values to insure that no 
    two image will be alike. This can, in most cases, be canged to a single 
    number for a constant value eq. angle = 15.

    Args:
        image (Image.Image): No Default
        [Pill image as input]\n
        density (tuple, optional):  Defaults to (0.03,0.14).
        [Density of the rain layer, as a probability of each pixel in low resolution space to be a rain drop.]\n
        density uniformity (tuple, optional): Defaults to (0.8,1.0).
        [Size uniformity of the raindrops. Higher values denote more similarly sized raindrops.]\n
        drop size (tuple, optional): Defaults to (0.3,0.4).
        [Size of the raindrop. This parameter controls the resolution at which raindrops are sampled.]\n
        drop size uniformity (tuple, optional): Defaults to (0.1,0.5).
        [Size uniformity of the raindrop. Higher values denote more similarly sized raindrops.]\n
        angle (tuple, optional):  Defaults to (-15,15).
        [Angle in degrees of motion blur applied to the raindrop]\n
        speed (tuple, optional): Defaults to (0.1,0.2).
        [This parameter controls the motion blur’s kernel size.]\n
        blur_sigma_fraction (tuple, optional): Defaults to (0.001, 0.001).
        [Standard deviation (as a fraction of the image size) of gaussian blur applied to the raindrops.]

    Returns:
        Image.Image: [Pill image]
    """
    noise_layer = iaa.RainLayer(
            density=density,
            density_uniformity=density_uniformity,
            drop_size=drop_size,
            drop_size_uniformity=drop_size_uniformity,
            angle=angle,
            speed=speed,
            blur_sigma_fraction=blur
    )
    images = np.asarray(image, dtype="uint32")
    img_aug = noise_layer.augment_image(images)

    pil_image = Image.fromarray(img_aug.astype('uint8'), 'RGB')
    return pil_image

def snow(image:Image.Image, density:tuple=(0.03,0.14), density_uniformity:tuple=(0.8,1.0), drop_size:tuple=(0.3,0.4), drop_size_uniformity:tuple=(0.1,0.5),angle:tuple=(-15,15), speed:tuple=(0.1,0.2), blur:tuple=(0.001, 0.001))->Image.Image: 
    """
    #TODO: Fine tune params
    Method to add snow particles to images. 
    The argurments are all tupples of values to insure that no 
    two image will be alike. This can, in most cases, be canged to a single 
    number for a constant value eq. angle = 15.

    Args:
        image (Image.Image): No Default
        [Pill image as input]\n
        density (tuple, optional):  Defaults to (0.03,0.14).
        [Density of the rain layer, as a probability of each pixel in low resolution space to be a rain drop.]\n
        density uniformity (tuple, optional): Defaults to (0.8,1.0).
        [Size uniformity of the snowflakes. Higher values denote more similarly sized snowflakes.]\n
        Flake size (tuple, optional): Defaults to (0.3,0.4).
        [Size of the snowflake. This parameter controls the resolution at which snowflakes are sampled.]\n
        Flake size uniformity (tuple, optional): Defaults to (0.1,0.5).
        [Size uniformity of the snowflake. Higher values denote more similarly sized snowflakes.]\n
        angle (tuple, optional):  Defaults to (-15,15).
        [Angle in degrees of motion blur applied to the snowflake]\n
        speed (tuple, optional): Defaults to (0.1,0.2).
        [This parameter controls the motion blur’s kernel size.]\n
        blur_sigma_fraction (tuple, optional): Defaults to (0.001, 0.001).
        [Standard deviation (as a fraction of the image size) of gaussian blur applied to the snowflakes.]

    Returns:
        Image.Image: [Pill image]
    """
    noise_layer = iaa.SnowflakesLayer(
            density=density,
            density_uniformity=density_uniformity,
            flake_size=drop_size,
            flake_size_uniformity=drop_size_uniformity,
            angle=angle,
            speed=speed,
            blur_sigma_fraction=blur
    )
    images = np.asarray(image, dtype="uint32")
    img_aug = noise_layer.augment_image(images)

    pil_image = Image.fromarray(img_aug.astype('uint8'), 'RGB')
    return pil_image

    images = np.asarray(image, dtype="uint8")
    img_aug = noise_layer.augment_image(images)

    pil_image = Image.fromarray(img_aug.astype('uint8'), 'RGB')
    return pil_image

def QuickDebug()-> None:
    # img = Image.open('Dataset/images/00000/00000_00029.ppm')
    # img = Fog(img)
    # img = img.save(f'C:/Users/roni/Desktop/rain/1.png')
    # n = [0.1, 0.3, 0.6, 1]
    for i in range(100):
        img = Image.open('C:/Users/jeppe/Desktop/FullIJCNN2013/00/00000.ppm')
        
    
        img = snow(img)
        img.show()
        #img.save(f'C:/Users/roni/Desktop/rain/{i}.png')
    # for i in range(0,50):
    #     img = Image.open('Dataset/images/00000/00000_00029.ppm')
    #     # img.show()
    
    #     img = Rain(img)
    #     img.save(f'C:/Users/roni/Desktop/rain/{i}.png')


QuickDebug()