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
    mode = 'rain'

    def __init__(self, config:dict):

        self.Keys = ['density','density_uniformity','drop_size','drop_size_uniformity','angle','speed','blur','mode']
        for key in self.Keys:
            if key in config:
                setattr(self, key, config.get(key))

    def rain(self,image:Image.Image)->Image.Image: 
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
                density=self.density,
                density_uniformity=self.density_uniformity,
                drop_size=self.drop_size,
                drop_size_uniformity=self.drop_size_uniformity,
                angle=self.angle,
                speed=self.speed,
                blur_sigma_fraction=self.blur
        )
        images = np.asarray(image, dtype="uint32")
        img_aug = noise_layer.augment_image(images)

        pil_image = Image.fromarray(img_aug.astype('uint8'), 'RGB')
        return pil_image

    def snow(self, image:Image.Image)->Image.Image: 
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
                density=self.density,
                density_uniformity=self.density_uniformity,
                flake_size=self.drop_size,
                flake_size_uniformity=self.drop_size_uniformity,
                angle=self.angle,
                speed=self.speed,
                blur_sigma_fraction=self.blur
        )
        images = np.asarray(image, dtype="uint32")
        img_aug = noise_layer.augment_image(images)

        pil_image = Image.fromarray(img_aug.astype('uint8'), 'RGB')
        return pil_image

        images = np.asarray(image, dtype="uint8")
        img_aug = noise_layer.augment_image(images)

        pil_image = Image.fromarray(img_aug.astype('uint8'), 'RGB')
        return pil_image

    def add_weather(self, img):
        if self.mode == 'rain':
            return self.rain(img)
        elif self.mode == 'snow':
            return self.snow(img)
        else:
            print(f'{self.mode} is not a valid specefication')

    def __add__(self, img):
       return self.add_weather(img)

def QuickDebug()-> None:
    img = Image.open('C:/Users/roni/Desktop/Project/BiksTurePy/Dataset/images/00000/00004_00010.ppm')
    config = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001)}
    w = weather(config)
    w.add_weather(img).show()


# QuickDebug()