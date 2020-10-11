import os,sys,inspect
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from General_image_func import changeImageSize,merge_two_images,convertToPILImg
from Noise_Generators.Main_Noise import Noise

def CreateDistination(dist):
    if(os.path.exists(dist)):
        pass
    else:
        os.mkdir(dist)
        
def loadImags(folder):
    loaded_img = []
    with os.scandir(folder) as imgs:
        for ppm_path in imgs:
            if ppm_path.name.endswith(".ppm"):
                loaded_img.append(Image.open(ppm_path.path))
    return loaded_img   

def CreateClassFolder(dist,folder,imgs):
    folder_Name = os.path.basename(os.path.normpath(folder))
    folder_Path = dist+"/"+folder_Name
    CreateDistination(folder_Path)
    for img in range(len(imgs)):
        imgs[img].save(folder_Path+"/"+str(img)+".ppm")


def AddNoiseToAllImg(folders,dist):
    CreateDistination(dist)
    for folder in folders:
        imgs = loadImags(folder)
        newImgs = []
        newImgs.extend(imgs)
        fog_set=(1)
        day_set=(0.5)
        wh_set = (70,7,2,(2,2),(130,130,130))
        for img in imgs:
            img = changeImageSize(100, 100, img)
            newImgs.append(Noise(img,fog_set=fog_set))
            newImgs.append(Noise(img,day_set=day_set))
            newImgs.append(Noise(img,wh_set=wh_set))
        CreateClassFolder(dist,folder,newImgs)
        del imgs #removes loaded images from memory

def Generate_Dataset(path,dist,resolution, mode = "all"):
    subfolders = [ f.path for f in os.scandir(path) if f.is_dir() ]
    if mode == "all":
        AddNoiseToAllImg(subfolders,dist)

path = 'C:/Users/jeppe/Desktop/FullIJCNN2013'
dist = "C:/Users/jeppe/Desktop/Test/pls"
Generate_Dataset(path,dist,1,mode = "all")