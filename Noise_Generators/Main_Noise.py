from AddWeather import add_rain
from Perlin_noise import Foggyfy
from birghtness import DayAdjustment
from PIL import Image

def Noise(img, fog_set = None, day_set = None, wh_set = None):
    #Input should be a pill image and tuples corrospondiong to the noises wanted.
    if(wh_set != None):
        img = add_rain(img, rain_drops=wh_set[0],drop_length=wh_set[1],drop_width=wh_set[2],blurr=wh_set[3],color=wh_set[4])
    
    if(fog_set != None):
        img = Foggyfy(img)
    
    if(day_set != None):
        img = DayAdjustment(img,day_set)
    
    return img

#img = Image.open(path)
#fog_set=(1)
#day_set=(0.5)
#wh_set = (70,7,2,(2,2),(130,130,130))
#Noise(img,fog_set = fog_set,day_set = day_set,wh_set = wh_set).show()
#Noise(img,day_set = (0.5)).show()
#Noise(img,day_set = (2.0)).show()
#Noise(img, wh_set = (70,7,2,(2,2),(130,130,130))).show()
#Noise(img,wh_set = (40,2,3,(3,3),(200,200,200))).show()
