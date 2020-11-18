import tkinter as tk
from PIL import ImageTk, Image
import os
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from Noise_Generators.homomorphic_filtering import homomorphic

image_path = ""

width = 300
height = 400

img_paths = "Dataset/ETSD_Adjusted"
roni_bot_path = "Dataset/roni_bot"

def img_count():
    global roni_bot_path
    
    return len(os.listdir(roni_bot_path))

def loadImags(folder):
    loaded_img = []
    with os.scandir(folder) as imgs:
        for ppm_path in imgs:
            if ppm_path.name.endswith(".ppm"):
                loaded_img.append(ppm_path.path)
    return loaded_img  

def load_X_images(path):
    subfolders = [ f.path for f in os.scandir(path) if f.is_dir() ]
    newImgs = []
    for folder in subfolders:
        imgs = loadImags(folder)
        imgs = [img for img in imgs]
        newImgs.extend(imgs)
    return newImgs

img_list = load_X_images(f"{img_paths}/Testing")
img_list.extend(load_X_images(f"{img_paths}/Training"))
img_index = 0

def next():
    global img_index, img_list
    img_index += 1
    try:
        return img_list[img_index]
    except Exception:
        raise Exception

def current():
    global img_index, img_list
    try:
        return img_list[img_index]
    except Exception:
        raise Exception

def previous():
    global img_index, img_list
    img_index = img_index - 1 if img_index > 0 else 0
    try:
        return img_list[img_index]
    except Exception:
        raise Exception

img_size = (200, 200)

def get_image(path):
    global img_size
    img = Image.open(path)
    img = img.resize(img_size)
    return img

top = tk.Tk()
canvas = tk.Canvas(top, width = width, height = height)  

first_path = next()

img = ImageTk.PhotoImage(get_image(first_path))
panel = tk.Label(top, image=img)
panel.pack(side="bottom", fill="both", expand="yes")

def reset_values():
    a.set(1)
    b.set(0)

def next_picure():
    img2 = ImageTk.PhotoImage(get_image(next()))
    panel.configure(image=img2)
    panel.image = img2
    reset_values()
    
def previous_picure():
    img2 = ImageTk.PhotoImage(get_image(previous()))
    panel.configure(image=img2)
    panel.image = img2
    reset_values()

canvas.pack()      

def generate_path(a, b):
    global img_index, roni_bot_path
    
    img_id = img_count()
    
    
    return f"{roni_bot_path}/{img_id}_a_{a}_b_{b}.ppm"

def callback():
    global img_list, img_index
    img = Image.open(current())
    img.save(generate_path(a.get(), b.get()))
    
    del img_list[img_index]
    next_picure()
    print (f"{a.get()} - {b.get()}")
    reset_values()

def trigger(g):
    global img_size
    config = {'a':a.get(),'b':b.get(),'cutoff':3}
    homo = homomorphic(config)
    img = homo.homofy(Image.open(current()))
    img = img.resize(img_size)
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img

a = tk.Scale(top, from_=0.0, to=2.0,orient=tk.HORIZONTAL, resolution=0.001, digits=4, length=300, command=trigger)
a.set(1)
a.pack()

b = tk.Scale(top, from_=0.0, to=2.0, orient=tk.HORIZONTAL, resolution=0.001, digits=4, length=300, command=trigger)
b.set(0)
b.pack()


bu = tk.Button(top, text="SAVE", command=callback)
bu.place(x=width/2, y = height-50)

ri = tk.Button(top, text="next", command=next_picure)
ri.place(x=width/2+100, y=height-50)

le = tk.Button(top, text="previous", command=previous_picure)
le.place(x=width/2-100, y=height-50)

top.mainloop()