from weather import weather
from perlin_noise import perlin
from brightness import brightness
from PIL import Image

class Filter:
    fog_set = None
    day_set = None
    wh_set = None
    def __init__(self,config:dict):
        Keys = ['fog_set','day_set','wh_set']
        if Keys[0] in config:
            self.fog_set = config.get(Keys[0])
        if config.get(Keys[1]) != None:
            self.day_set = config.get(Keys[1])
        if config.get(Keys[2]) != None:
            self.wh_set = config.get(Keys[2])
        
    def Apply(self,img):
        #Input should be a pill image and tuples corrospondiong to the noises wanted.
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

    def __add__(self,img):
        return self.Apply(img)

def QuickDebug():
    img = Image.open("C:\\Users\\jeppe\\Desktop\\GTSRB_Final_Training_Images\\GTSRB\\Final_Training\\Images\\00000\\00002_00029.ppm")

    p = {'octaves':6, 'persistence':0.5, 'lacunarity': 8, 'alpha': 0.5}
    day = {'factor':1.3} 
    night = {'factor':0.3}
    rain = {'rain_drops':200, 'drop_length':2,'drop_width':1,'blurr':(2,2),'color':(130,130,130)}

    Filter_Con = {'fog_set':p, 'day_set':day, 'wh_set':rain}
    F = Filter(Filter_Con)

    newImage = F+img
    newImage.show()

    """
    Noise(img,fog_set=p).show()


    Noise(img,day_set=day).show()
    Noise(img,day_set=night).show()

    Noise(img,wh_set=rain).show()
    """

QuickDebug()
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
