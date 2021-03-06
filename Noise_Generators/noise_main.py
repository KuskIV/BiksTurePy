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
from Noise_Generators.Premade_filters import get_premade_filter

class Filter:
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
                img = Image.fromarray((img * 255.0).astype(np.uint8))
                np_array_given = True
            img = func(self,img)
            if np_array_given:
                img = (np.asarray(img.convert('RGB')))/255
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
        """Returns a dictionary with the filter order

        Returns:
            [dict]: dict used to access the __init__ functions for the diffrent filters
        """
        return {self.Keys[2]:weather,
                self.Keys[0]:perlin,
                self.Keys[1]:brightness,
                self.Keys[4]:fog_remove, 
                self.Keys[3]:homomorphic}

    @check_adjust_img_type
    @decor_img_corretness
    def Filter_order(self,img:Image.Image)->Image.Image:
        """Method applying the filter in the correct order to the img

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
        for i in loading_bar(len(imgs)):
            returnList.append(self + imgs[i])
        return returnList
    
    def get_config(self):
        return self.configuration

def loading_bar(max:int):
    """loading bar, that can be applied easily in any loop

    Args:
        max (int): The value the loading bar count towards

    Yields:
        i (int): The currently reached value of i
    """
    done = max
    show_progress = True if max > 100 else False
    progress = trange(done, desc='mult stuff', leave=True) if show_progress else range(max)
    
    for i in progress:
        if show_progress:
            progress.set_description(f"{i}/{done} multi done")
            progress.refresh()
        yield i 

def chunk_it(seq, num):
    avg, out, last = len(seq) / float(num), [], 0.0
    while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg
    if(len(out) > num):
        out[-2] = seq[out[-2].start:out[-1].stop]
        del out[-1]
    return out

def linear_dist(images,lables,fil,chungus):
    result = []
    indexes=chunk_it(range(len(images)),len(fil) + chungus) 
    for i in range(len(indexes) - (chungus)):
        temp_list = fil[i][1]*images[indexes[i].start:indexes[i].stop]
        temptemp_list = lables[indexes[i].start:indexes[i].stop]
        for k in range(len(temp_list)):
            temp_list[k] = (temp_list[k],fil[i][0],temptemp_list[k])
        result.extend(temp_list)
    return result

def apply_multiple_filters(Imgs:list,mode = 'rand', KeepOriginal:bool=True, filters:dict=None, chungus=4)->list:
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
    result, fil, images, lables =[], list(filters.items()) ,Imgs[0], Imgs[1]

    if mode == 'linear':
        result = linear_dist(images, lables, fil, chungus)

    for i in range(len(images)):
        if KeepOriginal:
            result.append((images[i],'Original',lables[i]))
        if mode == 'rand':
            _tuple = random.choice(fil)
            result.append((_tuple[1]+images[i],_tuple[0],lables[i]))
    return result 

def premade_single_filter(str:str)->Filter:
    return Filter(get_premade_filter(str))

def load_level_of_filters(filter_name:str):
    filter_list = []
    for i in range(1, 11):
        filter_key = f'{filter_name}{i}'
        filter_input = f'mod_{filter_name}{i/10}'
        filter_list.append(premade_single_filter(filter_input))
    return filter_list
    
if __name__ == '__main__':
    pass
    # import time
    # path1 = 'C:/Users/jeppe/Downloads/Combnoise_for_coronoi-20201207T131136Z-001/Combnoise_for_coronoi/fog_night.png'
    # path2 = 'C:/Users/jeppe/Downloads/Combnoise_for_coronoi-20201207T131136Z-001/Combnoise_for_coronoi/rain_night1.png'
    # path3 = 'C:/Users/jeppe/Downloads/Combnoise_for_coronoi-20201207T131136Z-001/Combnoise_for_coronoi/snow_night.png'
    # path4 = 'C:/Users/jeppe/Downloads/Combnoise_for_coronoi-20201207T131136Z-001/Combnoise_for_coronoi/fog_rain10.png'
    # path5 = 'C:/Users/jeppe/Downloads/Combnoise_for_coronoi-20201207T131136Z-001/Combnoise_for_coronoi/fog_snow.png'
    path1 = 'C:/Users/jeppe/Desktop/GTSRB_Final_Training_Images/GTSRB/Final_Training/images/00014/00020_00029.ppm'
    save_path = 'C:/Users/jeppe/Desktop/Noise_levels'

    fog = load_level_of_filters("fog")
    fogs = [x+Image.open(path1) for x in fog]
    rain = load_level_of_filters("rain")
    rains = [x+Image.open(path1) for x in rain]
    snow = load_level_of_filters("snow")
    snows = [x+Image.open(path1) for x in snow]
    night = load_level_of_filters("night")
    nights = [x+Image.open(path1) for x in night]

    for i in range(len(fogs)):
        fogs[i].save(f"{save_path}/fog{i}.png")
        rains[i].save(f"{save_path}/rain{i}.png")
        snows[i].save(f"{save_path}/snow{i}.png")
        nights[i].save(f"{save_path}/night{i}.png")



    # name1 = "fog_night_homo_haze"
    # name2 = "fog_rain_homo_haze"
    # name3 = "fog_snow_homo_haze"
    # names = [name1,name2,name3]
    # alt1 = "fog_night"
    # alt2 = "fog_rain"
    # alt3 = "fog_snow"
    # alts = [alt1,alt2,alt3]
    # homo = premade_single_filter("std_homo")
    # filt1 = premade_single_filter("fog_night")
    # filt2 = premade_single_filter("fog_rain")
    # filt3 = premade_single_filter("fog_snow")
    # filters = [filt1,filt2,filt3]




    # for i in range(len(names)):

    #     img = Image.open(path1)
    #     img =(filters[i]+img)
    #     img.save(f"{save_path}/{alts[i]}.png")
    #     (homo+img).save(f"{save_path}/{names[i]}.png")
    # levels = ['mild','medium','heavy']
    # noises = ['rain','snow','night','fog']

    # # for level in levels:
    # #     for noise in noises:
    # #         filt = premade_single_filter(f"{noise}_{level}")
    # #         img = Image.open(path1)
    # #         (filt + img).save(f'{save_path}/{noise}_{level}.png')
    # filt = premade_single_filter("std_homo")
    # defilt = premade_single_filter("de_fog15")
    # img = Image.open(path1)
    # #img=(filt + img)
    # img.save(f'{save_path}/original.png')
    # (defilt+img).save(f'{save_path}/defog.png')
    # # filt_de_homo = premade_single_filter("dehaze_homo")

    # imgarr = []
    # for i in range(101):
    #     imgarr.append(Image.open(path1))
    # filt * imgarr
    # filters = {"fog":premade_single_filter("fog"),"rain":premade_single_filter("rain")}
    # imgarr = []
    # lalbes = []
    # for i in range(1000):
    #     imgarr.append(Image.open(path1))
    #     lalbes.append("yeet")
    # imgs = apply_multiple_filters((imgarr,lalbes),mode="linear",KeepOriginal=False,filters=filters, chungus=3)
    # for i in range(len(imgs)):
    #     imgs[i][0].save(f"{save_path}/{imgs[i][1]}_{i}.png")
    #filt * imgarr

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