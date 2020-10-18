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
from Models.test_model import accumilate_distribution
from global_paths import get_test_model_paths

lazy_split = 10
dataset_split = 0.7
folder_batch_size = 30

def get_ppm_arr(h5:h5py._hl.files.File, keys:list, ppm_name:str)->list:
    """Get the ppm array from h5 file

    Args:
        h5 (h5py._hl.files.File): h5py file that should be read
        keys (list): list of keys
        ppm_name (str): the name of the wanted ppm

    Returns:
        list: array represnting the ppm image form the h5py file
    """
    return h5[keys[0]][keys[1]][keys[2]][ppm_name]

def get_key(h5:h5py._hl.files.File, keys:list)->str:
    """Get the key for a image

    Args:
        h5 (h5py._hl.files.File): the h5py file
        keys (list): a list of the keys

    Returns:
        str: the key in string from
    """
    return h5[keys[0]][keys[1]][keys[2]]

def train_set_start_end(self, train_slice:int, current_iteration:int, is_last:bool, img_in_class:int)->tuple:
    #TODO pls delete mads
    start_val = train_slice * current_iteration
    end_val = train_slice * current_iteration + train_slice if not is_last else math.ceil(img_in_class * self.training_split)
    return start_val, end_val

def val_set_start_end(self, train_slice:int, val_slice:int, current_iteration:int, is_last:int, img_in_class:int, max_iteration:int)->tuple:
    #TODO pls delete mads FAST
    start_val = math.ceil(img_in_class * self.training_split) + (val_slice * current_iteration) if not is_last else math.ceil(img_in_class * self.training_split) + (max_iteration - 1) * (math.floor((img_in_class * (h5_object.get_val_size(self))) / max_iteration))
    end_val = math.ceil(img_in_class * self.training_split) + (val_slice * current_iteration + val_slice) if not is_last else img_in_class
    return start_val, end_val

def acc_dist_for_images(h5_obj:object)->None:
    train_images, train_labels, _, _ = h5_obj.shuffle_and_lazyload(0, 1)
    large_model_path, medium_model_path, small_model_path = get_test_model_paths()
    accumilate_distribution(large_model_path, train_images, train_labels)
    accumilate_distribution(medium_model_path, train_images, train_labels)
    accumilate_distribution(small_model_path, train_images, train_labels)


def find_ideal_model(h5_obj:object)->None:
    """finds the ideal model

    Args:
        h5_obj (object): h5 object
    """
    image_sizes = [(200, 200), (128, 128), (32, 32)]

    models = get_processed_models()

    train_images = []
    test_images = []

    for j in range(lazy_split):
        # generate models
        train_images, train_labels, test_images, test_labels = h5_obj.shuffle_and_lazyload(j, lazy_split)
        print(train_labels)
        print(f"Images in train_set: {len(train_images)} ({len(train_images) == len(train_labels)}), Images in val_set: {len(test_images)} ({len(test_images) == len(test_labels)})")
        print(f"This version will split the dataset in {lazy_split} sizes.")
        # zip together with its size
        model_and_size = list(zip(models, image_sizes))

        # train models
        for i in range(len(model_and_size)):
            print(f"Training model {i} / {len(model_and_size) - 1} for time {j} / {lazy_split - 1}")
            train_and_eval_models_for_size(models, model_and_size[i][1], model_and_size[i][0], i, train_images, train_labels, test_images, test_labels)

    large_model_path, medium_model_path, small_model_path = get_test_model_paths()

    store_model(models[0], large_model_path)
    store_model(models[1], medium_model_path)
    store_model(models[2], small_model_path)

if __name__ == "__main__":
    h5_obj = h5_object(folder_batch_size, get_key, get_ppm_arr, train_set_start_end, val_set_start_end, dataset_split)
    #find_ideal_model(h5_obj)
    acc_dist_for_images(h5_obj)
    
    # # This was a table generator for roni
    # h5_obj.print_class_data()