from Noise_Generators.AddWeather import AddParticels
from Noise_Generators.Perlin_noise import Foggyfy
from Noise_Generators.birghtness import DayAdjustment

def Noise(img, fog_set = None, day_set = None, wh_set = None):
    #Input should be a pill image and tuples corrospondiong to the noises wanted.
    
    if(fog_set != None):
        img = Foggyfy(img)
    
    if(day_set != None):
        img = DayAdjustment(img,day_set[0])
    
    if(wh_set != None):
        img = AddParticels(img, size=wh_set[0],Opacity=wh_set[1],frequency=wh_set[2],LoopJumpX=wh_set[3],LoopJumpY=wh_set[4])
    
    return img