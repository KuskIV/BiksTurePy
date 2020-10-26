

from PIL import Image
import random
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import global_paths
from general_image_func import changeImageSize
from Noise_Generators.weather_gen import weather
from Noise_Generators.Perlin_noise import perlin
from Noise_Generators.brightness import brightness

class Filter:
    """The filter class is a combination the three noises fog, 
    brightness and weather, the purpose is to provide a easy way the reuse and repeat the settings for a filter.
    And bmake them easier to iterface with in general.

    Returns:
        Image.Image : The returned image will have the filter configured to be added onto it.
    """
    #default values
    fog_set = None
    day_set = None
    wh_set = None
    def __init__(self,config:dict):
        """The to configure the default variables in perlin noise.

        Args:
            config (dict): The input should be a dictionary where the key is the name of the variable to change,
            The value would then be the value refresen by the keyword.
            Keys = ['fog_set','day_set','wh_set']
        """
        Keys = ['fog_set','day_set','wh_set']
        if Keys[0] in config:
            self.fog_set = config.get(Keys[0])
        if config.get(Keys[1]) != None:
            self.day_set = config.get(Keys[1])
        if config.get(Keys[2]) != None:
            self.wh_set = config.get(Keys[2])
        
    def Apply(self,img:Image.Image)->Image.Image:
        """The method that applies the filters onto the images

        Args:
            img (Image.Image): Normal open PIL image

        Returns:
            Image.Image: Image with filters on it
        """
        old_size = img.size
        img = changeImageSize(200,200,img)
        if(self.wh_set != None):
            wn =  weather(self.wh_set)
            img = wn.add_weather(img)
        
        if(self.fog_set != None):
            pn = perlin(self.fog_set)
            img = pn.Foggyfy(img)
        
        if(self.day_set != None):
            bn = brightness(self.day_set)
            img = bn.DayAdjustment(img)
        img = changeImageSize(old_size[0],old_size[1],img)
        return img

    def __add__(self,img:Image.Image)->Image.Image:
        """Interface method for the apply method to be able to wrtie more readable code.
        This enables the use of plus, so that a filter can be applied simply by adding the filter and image togther.

        Args:
            img (Image.Image): A Open PIL Image

        Returns:
            Image.Image: A picture with the applied filters
        """
        return self.Apply(img)

    def __mul__(self, imgs:list)->list:
        """Method for applying the same filter on multiple images

        Args:
            imgs (list): List of Pil images

        Returns:
            list: List of Pil images with the filter
        """
        returnList = []
        for img in imgs:
            returnList.append(self + img)
        return returnList

def normal_distribution(lst:list):
    mean = (len(lst) - 1) / 2
    stddev = len(lst) / 6
    while True:
        index = int(random.normalvariate(mean, stddev) + 0.5)
        if 0 <= index < len(lst):
            return lst[index]

def apply_multiple_filters(Imgs:list,mode = 'rand', KeepOriginal:bool=True, filters:dict=None, **kwargs)->list:
    """
    A function that takes a input of pictures and applys them to eahc picture based on the selected mode. 
    The result will contain the edited picture with the name of the filter useed and the class of the orignal picture. 
    Args:
        Imgs (list): A list of PIL images tupled with thier class
        mode (str, optional): the distribution mode, how should the diffrent noises be distributed. Defaults to 'rand'.
        KeepOriginal (bool, optional): Should the original image be keeped, in the returned list. Defaults to True.
        filters (dict, optional): dictionary of filter objectedf tupled with the name of the filter. Defaults to None.

    Returns:
        list: A list containing the image, tupled with the filter and its original class
    """
    result = []    
    if filters is None:
        filters = {}
        for key, value in kwargs.items():
            filters[key] = value

    fil = list(filters.items())
    for i in range(len(Imgs)):
        if KeepOriginal:
            result.append((Imgs[0][i],'Original'))
        if mode == 'rand':
            tuple = random.choice(fil)
            result.append((tuple[1]+Imgs[0][i],tuple[0]))
        
        if mode == 'normal':
            filter_and_lable = normal_distribution(fil)
            result.append((filter_and_lable[1]+Imgs[0][i],Imgs[1][i],filter_and_lable[0]))

    return result #(image,class,filter)

        
def loadImags(folder):
    loaded_img = []
    with os.scandir(folder) as imgs:
        for ppm_path in imgs:
            if ppm_path.name.endswith(".ppm"):
                loaded_img.append(Image.open(ppm_path.path))
    return loaded_img  

def load_X_images(path):
    subfolders = [ f.path for f in os.scandir(path) if f.is_dir() ]
    newImgs = []
    for folder in subfolders:
        imgs = loadImags(folder)
        imgs = [(img,os.path.basename(os.path.normpath(folder))) for img in imgs]
        newImgs.extend(imgs)
    return newImgs


def premade_single_filter(str:str)->Filter:
    config = {}
    if str == 'fog':
        config = {'octaves':8, 'persistence':0.3, 'lacunarity': 5, 'alpha': 0.4}
        result = Filter({'fog_set':config})
    if str == 'rain':
        config =     config = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001)}
        result = Filter({'wh_set':config})
    if str == 'snow':
        config =     config = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001),'mode':'snow'}
        result = Filter({'wh_set':config})
    if str == 'day':
        config = {'factor':1.0}
        result = Filter({'day_set':config})
    if str == 'night':
        config = {'factor':0.3}
        result = Filter({'day_set':config})
    return result

def QuickDebugL():
    #imgs = Image.open("C:\\Users\\jeppe\\Desktop\\GTSRB_Final_Training_Images\\GTSRB\\Final_Training\\Images\\00000\\00002_00029.ppm")
    imgs = load_X_images('C:/Users/jeppe/Desktop/FullIJCNN2013')
    F = premade_single_filter('fog')
    R = premade_single_filter('rain')
    S = premade_single_filter('snow')
    D = premade_single_filter('day')
    N = premade_single_filter('night')
    dict = {'fog':F,'rain':R,'snow':S,'day':D,'night':N}
    res = apply_multiple_filters(imgs,filters=dict, mode='normal', KeepOriginal=False)
    for i in range(len(res)):
        res[i][0].save(f'C:/Users/jeppe/Desktop/imagesFolder/{i}.png')
        

def QuickDebug():
    """Small debug function
    """
    img = Image.open('Dataset/images/00000/00000_00000.ppm')
    imgs = [img,img]
    p = {'octaves':8, 'persistence':0.8, 'lacunarity': 8, 'alpha': 0.4}
    day = {'factor':1.0} 
    night = {'factor':0.3}
    rain = {'rain_drops':500, 'drop_length':2,'drop_width':1,'blurr':(2,2),'color':(200,200,255)}

    Filter_Con = {'fog_set':p, 'day_set':day, 'wh_set':rain}
    F = Filter(Filter_Con)

    snow = premade_single_filter('fog')
    (F + img).show()
    #newImage = F*imgs
    #newImage[0].show()
    #newImage[1].show()

# QuickDebugL()
#fog_set=(1)
#day_set=(0.5)
#wh_set = (70,7,2,(2,2),(130,130,130))
#Noise(img,fog_set = fog_set,day_set = day_set,wh_set = wh_set).show()
#Noise(img,fog_set = (1)).save("C:/Users/jeppe/Desktop/Example_images/pic1.png")
#Noise(img,day_set = (0.5)).save("C:/Users/jeppe/Desktop/Example_images/pic2.png")
#Noise(img,day_set = (2.0)).save("C:/Users/jeppe/Desktop/Example_images/pic3.png")
#Noise(img, wh_set = (500,7,1,(2,2),(130,130,150))).save("C:/Users/jeppe/Desktop/Example_images/pic4.png")
#Noise(img,wh_set = (200,2,2,(5,5),(200,200,200))).save("C:/Users/jeppe/Desktop/Example_images/pic5.png")
#Noise(img,fog_set=(1), wh_set = (500,7,1,(2,2),(130,130,150))).save("C:/Users/jeppe/Desktop/Example_images/pic4.png")
