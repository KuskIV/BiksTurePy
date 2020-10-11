from Noise_Generators.Main_Noise import Noise
import os
from PIL import Image

def CreateDistination(dist):
    if(os.path.exists(dist)):
        pass
    else:
        os.mkdir(dist)
        
def loadImags(folder):
    loaded_img = []
    with os.scandir(folder) as imgs:
        for ppm_path in imgs:
            loaded_img.append(Image.open(ppm_path))
    return loaded_img   

def CreateClassFolder(dist,folder,imgs):
    folder_Name = os.path.basename(os.path.normpath(folder))
    folder_Path = dist+"/"+folder_Name
    CreateDistination(folder_Path)
    for img in range(imgs.count()):
        imgs[img].save(folder_Path+"/"+img+".ppm")


def AddNoiseToAllImg(folders,dist):
    CreateDistination(dist)
    for folder in folders:
        imgs = loadImags(folder)
        fog_set=(1)
        day_set=(0.5)
        wh_set = (70,7,2,(2,2),(130,130,130))
        for img in imgs:
            imgs.append(Noise(fog_set=fog_set))
            imgs.append(Noise(day_set=day_set))
            imgs.append(Noise(wh_set=wh_set))
        CreateClassFolder(dist,folder,imgs)
        del imgs #removes loaded images from memory

def Generate_Dataset(path,dist,resolution, mode = "all"):
    
    if mode == "all":
        AddNoiseToAllImg(path,dist)
