import numpy as np 
import seaborn as sbs
from load_h5 import h5_object
import matplotlib.pyplot as plt

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from global_paths import get_h5_train

def get_all_resolutions(h5_obj):
    resolutions = []
    for i in range(len(h5_obj.group)):
            keys = h5_obj.get_keys(h5_obj.group[i])
            if len(keys) > h5_obj.nested_level:
                for j in h5_obj.get_key(h5_obj.h5, keys):
                    arr = np.array(h5_obj.get_ppm_arr(h5_obj.h5, keys, j))
                    resolutions.append((arr.shape[0], arr.shape[1]))
    return resolutions

def average_ratio():
    pass

def average_resolution(h5_obj):
    resolutions = get_all_resolutions(h5_obj)
    
    img_size_products = [images[0] * images[1] for images in resolutions]
    img_size_ratios = [images[0] / images[1] for images in resolutions]

    sbs.set_theme()
    sbs.kdeplot(data=img_size_ratios)
    print("Plotting image size ratio")
    plt.show()

    sbs.kdeplot(data=img_size_products)
    print("Plotting image size product")
    plt.show()


# train_path = get_h5_train()
# h5_train = h5_object(train_path, training_split=1)
# average_resolution(h5_train)

# Average resolution is 82