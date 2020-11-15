
from PIL import Image
import random
import numpy as np
from tqdm import tqdm, trange
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import global_paths
from general_image_func import changeImageSize
from Noise_Generators.weather_gen import weather
from Noise_Generators.Perlin_noise import perlin
from Noise_Generators.brightness import brightness
from Noise_Generators.homomorphic_filtering import homomorphic

class Filter: #TODO create homophobic filter
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
    homo_set = None
    configuration = None

    def __init__(self,config:dict):
        """The to configure the default variables in perlin noise.

        Args:
            config (dict): The input should be a dictionary where the key is the name of the variable to change,
            The value would then be the value refresen by the keyword.
            Keys = ['fog_set','day_set','wh_set','homo_set']
        """
        self.configuration = config
        Keys = ['fog_set','day_set','wh_set','homo_set']
        if Keys[0] in config:
            self.fog_set = config.get(Keys[0])
        if config.get(Keys[1]) != None:
            self.day_set = config.get(Keys[1])
        if config.get(Keys[2]) != None:
            self.wh_set = config.get(Keys[2])
        if config.get(Keys[3]) != None:
            self.homo_set = config.get(Keys[3])
        
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

        if(self.homo_set != None):
            hom = homomorphic(self.homo_set)
            img = hom.homofy(img)

        img = changeImageSize(old_size[1],old_size[0],img)
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
        #TODO loading bar pls mads
        Args:
            imgs (list): List of Pil images

        Returns:
            list: List of Pil images with the filter
        """
        returnList = []

        done = len(imgs)
        progress = trange(done, desc='mult stuff', leave=True)

        for i in progress:
            progress.set_description(f"{i}/{done} multi done")
            progress.refresh()
            returnList.append(self + imgs[i])
        return returnList
        
    def get_config(self):
        return self.configuration

def normal_distribution(lst:list):
    mean = (len(lst) - 1) / 2
    stddev = len(lst) / 6
    while True:
        index = int(random.normalvariate(mean, stddev) + 0.5)
        if 0 <= index < len(lst):
            return lst[index]

def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

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

    lables = Imgs[1]
    images = Imgs[0]

    fil = list(filters.items())
    if mode == 'linear':
        indexes=chunk_it(range(len(images)),len(fil)+4) #TODO find ideal number to add here(didnt work)

        done = len(indexes)-4
        progress = trange(done, desc='index stuff', leave=True)        

        for i in progress:
            progress.set_description(f"{i+1} / {done} images has been processed")
            progress.refresh()

            for j in range(len(fil)):

                temp_list = fil[j][1]*images[indexes[i].start:indexes[i].stop]
                temptemp_list = lables[indexes[i].start:indexes[i].stop]
                for k in range(len(temp_list)):
                    temp_list[k] = (temp_list[k],fil[j][0],temptemp_list[k])
                    
                # lst = [(entry,fil[j][0],lables[0]) for entry in temp_list]
                
                result.extend(temp_list)

    done = len(lables)
    progress = trange(done, desc="Lable stuff", leave=True)
    
    for i in progress:
        progress.set_description(f"Image {i+1} / {done} has been processed")
        progress.refresh()

        if KeepOriginal:
            result.append((images[i],'Original',lables[i]))
        if mode == 'rand':
            _tuple = random.choice(fil)
            result.append((_tuple[1]+images[i],_tuple[0],lables[i]))
        
        if mode == 'normal':
            filter_and_lable = normal_distribution(fil)
            result.append((filter_and_lable[1]+images[i],lables[i],filter_and_lable[0]))

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
    newLables = []
    for folder in subfolders:
        imgs = loadImags(folder)
        imgs = [img for img in imgs]
        lables = [os.path.basename(os.path.normpath(folder)) for img in imgs]
        newImgs.extend(imgs)
        newLables.extend(lables)
    return (newImgs,newLables)


def premade_single_filter(str:str)->Filter:
    config = {}
    if str == 'fog':
        config = {'octaves':8, 'persistence':0.3, 'lacunarity': 5, 'alpha': 0.4}
        result = Filter({'fog_set':config})
    if str == 'fog_mild':
        config = {'octaves':1, 'persistence':0.2, 'lacunarity': 3, 'alpha': 0.5, 'darkness':0.5}
        result = Filter({'fog_set':config})
    if str == 'fog_medium':
        config = {'octaves':4, 'persistence':0.2, 'lacunarity': 3, 'alpha': 0.4, 'darkness':0.5}
        result = Filter({'fog_set':config})
    if str == 'fog_heavy':
        config = {'octaves':4, 'persistence':0.2, 'lacunarity': 3, 'alpha': 0.1, 'darkness':0.5}
        result = Filter({'fog_set':config})
    if str == 'rain':
        config =     config = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001)}
        result = Filter({'wh_set':config})
    if str == 'rain_mild':
        config =     config = {'density':(0.01,0.02),'density_uniformity':(0.7,1.0),'drop_size':(0.25,0.3),'drop_size_uniformity':(0.1,0.5),'angle':(-20,20),'speed':(0.1,0.15),'blur':(0.004,0.01)}
        result = Filter({'wh_set':config})
    if str == 'rain_medium':
        config =     config = {'density':(0.06,0.08),'density_uniformity':(0.7,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-20,20),'speed':(0.1,0.15),'blur':(0.001,0.001)}
        result = Filter({'wh_set':config})
    if str == 'rain_heavy':
        config =     config = {'density':(0.1,0.15),'density_uniformity':(0.9,1.0),'drop_size':(0.5,0.65),'drop_size_uniformity':(0.1,0.5),'angle':(-20,20),'speed':(0.1,0.2),'blur':(0.001,0.001)}
        result = Filter({'wh_set':config})
    if str == 'snow':
        config =     config = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001),'mode':'snow'}
        result = Filter({'wh_set':config})
    if str == 'snow_mild':
        config =     config = {'density':(0.1,0.15),'density_uniformity':(0.95,1.0),'drop_size':(0.8,0.9),'drop_size_uniformity':(0.2,0.6),'angle':(-30,30),'speed':(0.1,0.15),'blur':(0.004,0.01),'mode':'snow'}
        result = Filter({'wh_set':config})
    if str == 'snow_medium':
        config =     config = {'density':(0.2,0.27),'density_uniformity':(0.95,1.0),'drop_size':(0.8,0.9),'drop_size_uniformity':(0.2,0.6),'angle':(-30,30),'speed':(0.1,0.15),'blur':(0.004,0.01),'mode':'snow'}
        result = Filter({'wh_set':config})
    if str == 'snow_heavy':
        config =     config = {'density':(0.35,0.45),'density_uniformity':(0.95,1.0),'drop_size':(0.8,0.9),'drop_size_uniformity':(0.2,0.6),'angle':(-30,30),'speed':(0.1,0.15),'blur':(0.004,0.01),'mode':'snow'}
        result = Filter({'wh_set':config})
    if str == 'day':
        config = {'factor':1.0}
        result = Filter({'day_set':config})
    if str == 'night':
        config = {'factor':0.3}
        result = Filter({'day_set':config})
    if str == 'night_mild':
        config = {'factor':0.5}
        result = Filter({'day_set':config})
    if str == 'night_medium':
        config = {'factor':0.3}
        result = Filter({'day_set':config})
    if str == 'night_heavy':
        config = {'factor':0.15}
        result = Filter({'day_set':config})
    if str == 'std_homo':
        config = {'a':0.5,'b':0.9,'cutoff':3}
        result = Filter({'homo_set':config})
    return result

def QuickDebugL():
    #imgs = Image.open("C:\\Users\\jeppe\\Desktop\\GTSRB_Final_Training_Images\\GTSRB\\Final_Training\\Images\\00000\\00002_00029.ppm")
    imgs = load_X_images('C:/Users/jeppe/Desktop/FullIJCNN2013')
    F = premade_single_filter('fog')
    R = premade_single_filter('rain')
    S = premade_single_filter('snow')
    D = premade_single_filter('day')
    N = premade_single_filter('night')
    homo = premade_single_filter('std_homo')
    dict = {'fog':F,'rain':R,'snow':S,'day':D,'night':N}
    homom = {'homo':homo}
    res = apply_multiple_filters(imgs,filters=homom, mode='linear', KeepOriginal=False)
    for i in range(len(res)):
        res[i][0].save(f'C:/Users/jeppe/Desktop/Coroni_wrong/{i}.png')
        

def QuickDebug():
    """Small debug function
    """
    img = Image.open('E:/satina_gains_images/satina_tests/13/G_11415.ppm')
    imgs = [img,img]
    fog = {'octaves':8, 'persistence':0.3, 'lacunarity': 5, 'alpha': 0.4}
    day = {'factor':1.0} 
    night = {'factor':0.3}
    #snow = {'density':(0.2,0.3),'density_uniformity':(0.95,1.0),'drop_size':(0.8,0.9), 'drop_size_uniformity':(0.2,0.6),'angle':(-30,30),'speed':(0.08,0.15),'blur':(0.004,0.01)}
    #snow = {'density':(0.1,0.15),'density_uniformity':(0.95,1.0),'drop_size':(0.8,0.9),'drop_size_uniformity':(0.2,0.6),'angle':(-30,30),'speed':(0.1,0.15),'blur':(0.004,0.01)}
    # rain = {}
    #rain_mild["density"] = (0.5, 1)

    Filter_Con = {'fog_set':fog}
    F = Filter(Filter_Con)

    fog = premade_single_filter('fog')
    #(F + img).show()
    (F + img).save('C:/Users/jeppe/Desktop/Coroni_wrong/fog.png')
    #newImage = F*imgs
    #newImage[0].show()
    #newImage[1].show()

if __name__ == '__main__':
    QuickDebugL()
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
