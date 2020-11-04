import numpy as np
import seaborn as sbs
from Dataset.load_h5 import h5_object
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
import os

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from global_paths import get_h5_train, get_h5_test

def get_all_resolutions(h5_obj):
    resolutions = []

    done_len = len(h5_obj.group)
    progress = trange(done_len, desc='Sea stuff', leave=True)

    for i in progress:
        progress.set_description(f"Group {i+1} / {done_len} has been added")
        progress.refresh()

        keys = h5_obj.get_keys(h5_obj.group[i])
        if len(keys) > h5_obj.nested_level:
            for j in h5_obj.get_key(h5_obj.h5, keys):
                arr = np.array(h5_obj.get_ppm_arr(h5_obj.h5, keys, j))
                resolutions.append((arr.shape[0], arr.shape[1]))

    return resolutions

def average_ratio():
    pass

def average_resolution(lo):
    resolutions = lo

    img_size_products = [images[0] * images[1] for images in resolutions]
    img_size_ratios = [images[0] / images[1] for images in resolutions]

    sbs.set_theme()

    r_label = "Image ratio"
    imgs_r_d = {r_label:img_size_ratios}

    sbs.kdeplot(data=imgs_r_d, x=r_label)
    print("Plotting image size ratio")
    plt.show()

    p_label = "Product size for images (w*h)"
    imgs_p_d = {p_label:img_size_products}

    sbs.kdeplot(data=imgs_p_d, x=p_label)
    print("Plotting image size product")
    plt.show()

train_path = get_h5_train()
test_path = get_h5_test()
h5_train = h5_object(train_path, training_split=1)
h5_test = h5_object(test_path, training_split=1)

full_dataset_res = get_all_resolutions(h5_train)
full_dataset_res.extend(get_all_resolutions(h5_test))

print(len(full_dataset_res))
print(full_dataset_res[0])

average_resolution(full_dataset_res)


# Average resolution is 82
