from find_ideal_model import get_processed_models, train_and_eval_models_for_size
import numpy as np
from PIL import Image
import math
import h5py

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from Dataset.load_h5 import h5_object
from Models.create_model import store_model

lazy_split = 10
dataset_split = 0.7
folder_batch_size = 30

def get_ppm_arr(h5:h5py._hl.files.File, keys:list, ppm_name:str)->str:
    return h5[keys[0]][keys[1]][keys[2]][ppm_name]

def get_key(h5:h5py._hl.files.File, keys:list)->str:
    return h5[keys[0]][keys[1]][keys[2]]

def train_set_start_end(self, train_slice:int, current_iteration:int, is_last:bool, img_in_class:int)->tuple:
    start_val = train_slice * current_iteration
    end_val = train_slice * current_iteration + train_slice if not is_last else math.ceil(img_in_class * self.training_split)
    return start_val, end_val

def val_set_start_end(self, train_slice:int, val_slice:int, current_iteration:int, is_last:int, img_in_class:int, max_iteration:int)->tuple:
    start_val = math.ceil(img_in_class * self.training_split) + (val_slice * current_iteration) if not is_last else math.ceil(img_in_class * self.training_split) + (max_iteration - 1) * (math.floor((img_in_class * (h5_object.get_val_size(self))) / max_iteration))
    end_val = math.ceil(img_in_class * self.training_split) + (val_slice * current_iteration + val_slice) if not is_last else img_in_class
    return start_val, end_val

def find_ideal_model(h5_obj:object)->None:
    image_sizes = [(200, 200), (128, 128), (32, 32)]

    models = get_processed_models()

    for j in range(lazy_split):

        # generate models
        train_images, train_labels, test_images, test_labels = h5_obj.lazyload_h5(j, lazy_split)
        print(f"Images in train_set: {len(train_images)} ({len(train_images) == len(train_labels)}), Images in val_set: {len(test_images)} ({len(test_images) == len(test_labels)})")
        print(f"This version will split the dataset in {lazy_split} sizes.")

        # zip together with its size
        model_and_size = list(zip(models, image_sizes))

        # train models
        for i in range(len(model_and_size)):
            print(f"Training model {i} / {len(model_and_size) - 1} for time {j} / {lazy_split - 1}")
            train_and_eval_models_for_size(models, model_and_size[i][1], model_and_size[i][0], i, train_images, train_labels, test_images, test_labels)

    store_model(models[0], "large200")
    store_model(models[1], "medium128")
    store_model(models[2], "default32")

if __name__ == "__main__":
    h5_obj = h5_object(folder_batch_size, get_key, get_ppm_arr, train_set_start_end, val_set_start_end, dataset_split)
    #find_ideal_model(h5_obj)
    
    # # This was a table generator for roni
    # h5_obj.print_class_data()