import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from src.Noise_Generators.weather import weather
from src.Noise_Generators.Perlin_noise import perlin
from src.Noise_Generators.brightness import brightness
from PIL import Image

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
        if(self.wh_set != None):
            wn =  weather(self.wh_set)
            img = wn.add_rain(img)
        
        if(self.fog_set != None):
            pn = perlin(self.fog_set)
            img = pn.Foggyfy(img)
        
        if(self.day_set != None):
            bn = brightness(self.day_set)
            img = bn.DayAdjustment(img)
        
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
        returnLiSt = []
        for img in imgs:
            returnLiSt.append(self + img)
        return returnLiSt
def premade_single_filter(str:str)->Filter:
    config = {}
    if str == 'fog':
        config = {'octaves':8, 'persistence':0.3, 'lacunarity': 5, 'alpha': 0.4}
        result = Filter({'fog_set':config})
    if str == 'rain':
        config = {'rain_drops':500, 'drop_length':2,'drop_width':1,'blurr':(2,2),'color':(200,200,255)}
        result = Filter({'wh_set':config})
    if str == 'snow':
        config = {'rain_drops':700, 'drop_length':2,'drop_width':2,'blurr':(2,2),'color':(255,255,255)}
        result = Filter({'wh_set':config})
    if str == 'day':
        config = {'factor':1.0}
        result = Filter({'day_set':config})
    if str == 'night':
        config = {'factor':0.3}
        result = Filter({'day_set':config})
    return result

def QuickDebug():
    """Small debug function
    """
    img = Image.open("C:\\Users\\jeppe\\Desktop\\GTSRB_Final_Training_Images\\GTSRB\\Final_Training\\Images\\00000\\00002_00029.ppm")
    imgs = [img,img]
    p = {'octaves':8, 'persistence':0.8, 'lacunarity': 8, 'alpha': 0.4}
    day = {'factor':1.0} 
    night = {'factor':0.3}
    rain = {'rain_drops':500, 'drop_length':2,'drop_width':1,'blurr':(2,2),'color':(200,200,255)}

    Filter_Con = {'fog_set':p, 'day_set':day, 'wh_set':rain}
    F = Filter(Filter_Con)

    snow = premade_single_filter('fog')
    (snow + img).show()
    #newImage = F*imgs
    #newImage[0].show()
    #newImage[1].show()

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
