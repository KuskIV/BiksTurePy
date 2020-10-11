from Noise_Generators.AddWeather import add_rain
from Noise_Generators.Perlin_noise import Foggyfy
from Noise_Generators.birghtness import DayAdjustment
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

#img = Image.open("C:\\Users\\jeppe\\Desktop\\GTSRB_Final_Training_Images\\GTSRB\\Final_Training\\Images\\00000\\00002_00029.ppm")
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
