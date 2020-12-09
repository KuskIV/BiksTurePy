import re
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
from Noise_Generators.fog_remove import fog_remove

class Filter: #TODO create homophobic filter
    """The filter class is a combination the three noises fog, 
    brightness and weather, the purpose is to provide a easy way the reuse and repeat the settings for a filter.
    And bmake them easier to iterface with in general.

    Returns:
        Image.Image : The returned image will have the filter configured to be added onto it.
    """
    #default values
    fog_set = day_set = wh_set = homo_set = defog_set = configuration = None

    def __init__(self,config:dict):
        """The to configure the default variables in perlin noise.

        Args:
            config (dict): The input should be a dictionary where the key is the name of the variable to change,
            The value would then be the value refresen by the keyword.
            Keys = ['fog_set','day_set','wh_set','homo_set','defog_set']
        """
        self.configuration = config
        self.Keys = ['fog_set','day_set','wh_set','homo_set','defog_set']
        for key in self.Keys:
            if key in config:
                setattr(self, key, config.get(key))

    def check_adjust_img_type(func):
        """Checks what type of image is given, then converts that to a PIL image

        Args:
            func (function): The function that is decorated
        """
        def inner_func1(self,img):
            np_array_given = False
            if isinstance(img, np.ndarray):
                img = img * 255.0
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
                np_array_given = True
            img = func(self,img)
            if np_array_given:
                img=np.asarray(img.convert('RGB'))
                img = img / 255.0
                np_array_given = False
            return img
        return inner_func1

    def decor_img_corretness(func):
        """Decorator method that ensures that the image is the corret size for the filters.
        Args:
            func (function): The function this is used on
        """
        def inner_func(self,img):
            old_size = img.size
            img = changeImageSize(200,200,img)
            img = func(self,img)
            img = changeImageSize(old_size[1],old_size[0],img)
            return img
        return inner_func

    def get_order(self):
        return {self.Keys[2]:weather,
                self.Keys[0]:perlin,
                self.Keys[1]:brightness,
                self.Keys[4]:fog_remove, 
                self.Keys[3]:homomorphic}
    @check_adjust_img_type
    @decor_img_corretness
    def Filter_order(self,img:Image.Image)->Image.Image:
        """Method dictating the order of the filter when applied to some image

        Args:
            img (PIL.image): a Pil image that the diffrent filter should be applied to

        Returns:
            img (Pil.image): The image with the noise added
        """
        order = self.get_order()
        for key in self.Keys:
            if getattr(self, key) != None:
                obj = order[key](getattr(self, key))
                img = obj + img
        return img

    def Apply(self,img:Image.Image)->Image.Image:
        """The method that applies the filters onto the images

        Args:
            img (Image.Image): Normal open PIL image

        Returns:
            Image.Image: Image with filters on it
        """
        img = self.Filter_order(img) 
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

        if len(imgs) > 100:
            done = len(imgs)
            show_progress = True
            progress = trange(done, desc='mult stuff', leave=True)
        else:
            show_progress = False
            progress = range(len(imgs))
        
        try:
            for i in progress:
                if show_progress:
                    progress.set_description(f"{i}/{done} multi done")
                    progress.refresh()
                returnList.append(self + imgs[i])
            return returnList
        except Exception as e:
            print(f"ERROR: {e}")
            raise Exception
        
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

def apply_multiple_filters(Imgs:list,mode = 'rand', KeepOriginal:bool=True, filters:dict=None, chungus=4, **kwargs)->list:
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
        indexes=chunk_it(range(len(images)),len(fil) + chungus) #TODO find ideal number to add here(didnt work)

        done = len(indexes) - chungus
        progress = trange(done, desc='index stuff', leave=True)        
        try:
            for i in progress:
                try:
                    progress.set_description(f"{i+1} / {done} chunks has been processed")
                    progress.refresh()
                except Exception as e:
                    print(f"ERROR: {e}, {done}")
    
                # for j in range(len(fil)):

                temp_list = fil[i][1]*images[indexes[i].start:indexes[i].stop]
                temptemp_list = lables[indexes[i].start:indexes[i].stop]
                for k in range(len(temp_list)):
                    temp_list[k] = (temp_list[k],fil[i][0],temptemp_list[k])
                    
                # lst = [(entry,fil[j][0],lables[0]) for entry in temp_list]
                
                result.extend(temp_list)
        except Exception as e:
            print(f"ERROR: {e}")
            raise Exception

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
    imgName = []
    with os.scandir(folder) as imgs:
        for ppm_path in imgs:
            if ppm_path.name.endswith(".ppm"):
                loaded_img.append(Image.open(ppm_path.path))
                imgName.append(ppm_path.name.split(".")[0])
    return loaded_img, imgName

def load_X_images(path):
    subfolders = [ f.path for f in os.scandir(path) if f.is_dir() ]
    imgName = []
    newImgs = []
    newLables = []
    for folder in subfolders:
        imgs, nwimgName = loadImags(folder)
        imgName.extend(nwimgName)
        imgs = [img for img in imgs]
        lables = [os.path.basename(os.path.normpath(folder)) for img in imgs]
        newImgs.extend(imgs)
        newLables.extend(lables)

    return (newImgs,newLables,imgName)


def premade_single_filter(str:str)->Filter:
    config = {}

    #special case fog with ajustable aplha
    if "mod" in str:
        if not re.search("mod_fog\d+\.\d+",str) == None:
            modifier = re.search("\d+\.\d+",str)
            config = {'octaves':4, 'persistence':0.2, 'lacunarity': 3, 'alpha': 1-0.6*(float(modifier.group(0))), 'darkness':0.5}
            result = Filter({'fog_set':config})

        if not re.search("mod_night\d+\.\d+", str) == None:
            modifier = re.search("\d+\.\d+",str)
            config = {'factor': 1-0.8*(float(modifier.group(0))-0.001)}
            homo = {'a':1,'b':0.5,'cutoff':800}
            result = Filter({'day_set':config,'homo_set':homo})
            # result = Filter({'day_set':config})

        if not re.search("mod_rain\d+\.\d+",str) == None:
            modifier = re.search("\d+\.\d+",str)
            config ={'density':(0.1*float(modifier.group(0)), 0.15*float(modifier.group(0))),'density_uniformity':(0.9,1.0),'drop_size':(0.5,0.65),'drop_size_uniformity':(0.1,0.5),'angle':(-20,20),'speed':(0.1,0.2),'blur':(0.001,0.001)}
            result = Filter({'wh_set':config})

        if not re.search("mod_snow\d+\.\d+", str) == None:
            modifier = re.search("\d+\.\d+", str)
            config = {'density':(0.11*float(modifier.group(0)),0.16*float(modifier.group(0))),'density_uniformity':(0.95,1.0),'drop_size':(0.7,0.8),'drop_size_uniformity':(0.2,0.3),'angle':(-30,30),'speed':(0.04,0.1),'blur':(0.004,0.01),'mode':'snow'}
            result = Filter({'wh_set':config})

    if str == 'fog':
        config = {'octaves':4, 'persistence':0.2, 'lacunarity': 3, 'alpha': 0.4, 'darkness':0.5}
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
        config =  {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001)}
        result = Filter({'wh_set':config})
    if str == 'rain_mild':
        config = {'density':(0.01,0.02),'density_uniformity':(0.7,1.0),'drop_size':(0.25,0.3),'drop_size_uniformity':(0.1,0.5),'angle':(-20,20),'speed':(0.1,0.15),'blur':(0.004,0.01)}
        result = Filter({'wh_set':config})
    if str == 'rain_medium':
        config ={'density':(0.06,0.08),'density_uniformity':(0.7,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-20,20),'speed':(0.1,0.15),'blur':(0.001,0.001)}
        result = Filter({'wh_set':config})
    if str == 'rain_heavy':
        config ={'density':(0.1,0.15),'density_uniformity':(0.9,1.0),'drop_size':(0.5,0.65),'drop_size_uniformity':(0.1,0.5),'angle':(-20,20),'speed':(0.1,0.2),'blur':(0.001,0.001)}
        result = Filter({'wh_set':config})
    if str == 'snow':
        config = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.7,0.8),'drop_size_uniformity':(0.2,0.3),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001),'mode':'snow'}
        result = Filter({'wh_set':config})
    if str == 'snow_mild':
        config ={'density':(0.07,0.07),'density_uniformity':(0.95,1.0),'drop_size':(0.7,0.8),'drop_size_uniformity':(0.2,0.3),'angle':(-30,30),'speed':(0.04,0.1),'blur':(0.004,0.01),'mode':'snow'}
        result = Filter({'wh_set':config})
    if str == 'snow_medium':
        config = {'density':(0.09,0.1),'density_uniformity':(0.95,1.0),'drop_size':(0.7,0.8),'drop_size_uniformity':(0.2,0.3),'angle':(-30,30),'speed':(0.04,0.1),'blur':(0.004,0.01),'mode':'snow'}
        result = Filter({'wh_set':config})
    if str == 'snow_heavy':
        config = {'density':(0.11,0.16),'density_uniformity':(0.95,1.0),'drop_size':(0.7,0.8),'drop_size_uniformity':(0.2,0.3),'angle':(-30,30),'speed':(0.04,0.1),'blur':(0.004,0.01),'mode':'snow'}
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
        config = {'a':1,'b':0.5,'cutoff':800}
        result = Filter({'homo_set':config})
    if str == 'foghomo':
        config_h = {'a':1,'b':0.5,'cutoff':800}
        config_f = {'octaves':8, 'persistence':0.3, 'lacunarity': 5, 'alpha': 0.4}
        result = Filter({'fog_set':config_f,'homo_set':config_h})
    if str == 'rainhomo':
        config_h = {'a':1,'b':0.5,'cutoff':800}
        config_r= {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001)}
        result = Filter({'wh_set':config_r,'homo_set':config_h})
    if str == 'snowhomo':
        config_h = {'a':1,'b':0.5,'cutoff':800}
        config_s = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.7,0.8),'drop_size_uniformity':(0.2,0.3),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001),'mode':'snow'}
        result = Filter({'wh_set':config_s,'homo_set':config_h})
    if str == 'dayhomo':
        config_h = {'a':1,'b':0.5,'cutoff':800}
        config_d = {'factor':1.0}
        result = Filter({'day_set':config_d,'homo_set':config_h})
    if str == 'nighthomo':
        config_h = {'a':1,'b':0.5,'cutoff':800}
        config_n = {'factor':0.3}
        result = Filter({'day_set':config_n,'homo_set':config_h})
    if str == 'fog_night':
        config_f = {'octaves':8, 'persistence':0.3, 'lacunarity': 5, 'alpha': 0.4}
        config_n = {'factor':0.3}
        result = Filter({'day_set':config_n,'fog_set':config_f})
    if str == 'fog_day':
        config_f = {'octaves':8, 'persistence':0.3, 'lacunarity': 5, 'alpha': 0.4}
        config_n = {'factor':1.0}
        result = Filter({'day_set':config_n,'fog_set':config_f})
    if str == 'fog_rain':
        config_f = {'octaves':8, 'persistence':0.3, 'lacunarity': 5, 'alpha': 0.4}
        config_w = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001),'mode':'snow'}
        result = Filter({'wh_set':config_w,'fog_set':config_f})
    if str == 'fog_snow':
        config_f = {'octaves':8, 'persistence':0.3, 'lacunarity': 5, 'alpha': 0.4}
        config_w = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.7,0.8),'drop_size_uniformity':(0.2,0.3),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001),'mode':'snow'}
        result = Filter({'wh_set':config_w,'fog_set':config_f})
    if str == 'rain_night':
        config_w = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001),'mode':'snow'}
        config_n = {'factor':0.3}
        result = Filter({'wh_set':config_w,'day_set':config_n})
    if str == 'snow_night':
        config_w = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.7,0.8),'drop_size_uniformity':(0.2,0.3),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001),'mode':'snow'}
        config_n = {'factor':0.3}
        result = Filter({'wh_set':config_w,'day_set':config_n})
    if str == 'de_fog15':
        config = {'kernel':15}
        result = Filter({'defog_set':config})
    if str == 'de_fog10':
        config = {'kernel':10}
        result = Filter({'defog_set':config}) 
    if str == 'de_fog5':
        config = {'kernel':5}
        result = Filter({'defog_set':config})
    if str == 'fog_dehaze':
        config_f = {'octaves':4, 'persistence':0.2, 'lacunarity': 3, 'alpha': 0.5, 'darkness':0.5}
        config_d = {'kernel':15}
        result = Filter({'fog_set':config_f,'defog_set':config_d})
    if str == 'rain_dehaze':
        config_r = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001)}
        config_d = {'kernel':15}
        result = Filter({'wh_set':config_r,'defog_set':config_d})
    if str == 'snow_dehaze':
        config_s = {'density':(0.03,0.14),'density_uniformity':(0.8,1.0),'drop_size':(0.3,0.4),'drop_size_uniformity':(0.1,0.5),'angle':(-15,15),'speed':(0.1,0.2),'blur':(0.001,0.001)}
        config_d = {'kernel':15}
        result = Filter({'wh_set':config_s,'defog_set':config_d})
    if str == 'night_dehaze':
        config_n = {'factor':0.5}
        config_d = {'kernel':15}
        result = Filter({'day_set':config_n,'defog_set':config_d})
    if str == 'day_dehaze':
        config_da = {'factor':1.0}
        config_d = {'kernel':15}
        result = Filter({'day_set':config_da,'defog_set':config_d})
    if str == 'dehaze_homo':
        config_d = {'kernel':15}
        config = {'a':1,'b':0.5,'cutoff':800}
        result = Filter({'homo_set':config,'defog_set':config_d})
        
    if result == None:
        raise KeyError(f"(noise main) No such key exists: {str}")
    
    return result

if __name__ == '__main__':
    # import time
    # path1 = 'C:/Users/jeppe/Downloads/Combnoise_for_coronoi-20201207T131136Z-001/Combnoise_for_coronoi/fog_night.png'
    # path2 = 'C:/Users/jeppe/Downloads/Combnoise_for_coronoi-20201207T131136Z-001/Combnoise_for_coronoi/rain_night1.png'
    # path3 = 'C:/Users/jeppe/Downloads/Combnoise_for_coronoi-20201207T131136Z-001/Combnoise_for_coronoi/snow_night.png'
    # path4 = 'C:/Users/jeppe/Downloads/Combnoise_for_coronoi-20201207T131136Z-001/Combnoise_for_coronoi/fog_rain10.png'
    # path5 = 'C:/Users/jeppe/Downloads/Combnoise_for_coronoi-20201207T131136Z-001/Combnoise_for_coronoi/fog_snow.png'
    # path1 = 'C:/Users/jeppe/Desktop/GTSRB_Final_Training_Images/GTSRB/Final_Training/images/00014/00020_00029.ppm'
    # save_path = 'C:/Users/jeppe/Desktop/Noise_levels'
    # levels = ['mild','medium','heavy']
    # noises = ['rain','snow','night','fog']

    # for level in levels:
    #     for noise in noises:
    #         filt = premade_single_filter(f"{noise}_{level}")
    #         img = Image.open(path1)
    #         (filt + img).save(f'{save_path}/{noise}_{level}.png')
    # filt = premade_single_filter("fog_night")
    # img = Image.open(path1)
    # (filt + img).save(f'{save_path}/fog_night.png')
    # filt_de_homo = premade_single_filter("dehaze_homo")
    # img1 = Image.open(path1)
    # img2 = Image.open(path1)
    # img3 = Image.open(path1)
    # img4 = Image.open(path4)
    # img5 = Image.open(path5)

    # img1 = filt_homo + img1
    # img2 = filt_homo + img2
    # img3 = filt_homo + img3
    # img4 = filt_de_homo + img4
    # img5 = filt_de_homo + img5

    # img1.show()
    # img1.save(save_path+"/fog_night_homo_haze.png")
    # img2.save(save_path+"/rain_night_homo.png")
    # img3.save(save_path+"/snow_night_homo.png")
    # img4.save(save_path+"/fog_rain_homo_haze.png")
    # img5.save(save_path+"/fog_snow_homo_haze.png")
    #QuickDebug()
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
    