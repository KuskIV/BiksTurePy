import h5py
import numpy as np
import os
import io
from PIL import Image
import math
from os import path
import os.path
import sys
import re
from PIL import Image
import random

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from general_image_func import convertToPILImg
from global_paths import get_h5_path

data = []
group = []

def sort_groups(name, obj):
    if isinstance(obj, h5py.Dataset):
        data.append(name)
    elif isinstance(obj, h5py.Group):
        group.append(name)

def get_h5(h5_path):
    if not path.exists(h5_path):
        print(f"The path for the h5 file does not exist ({h5_path}). The program has exited.")
        sys.exit()
    else:
        h5 = h5py.File(h5_path, 'r')
        h5.visititems(sort_groups)
        return h5


## DONT DELETE THESE THREE LINES OF CODE!:
#img_as_arr = np.array(self.h5[keys[0]][keys[1]][ppm_names[i]])
#img = Image.fromarray(img_as_arr.astype('uint8'), 'RGB')
#img.show()

def Shuffle(img_dataset, img_labels):
    img_dataset_in = img_dataset
    img_labels_in = img_labels

    z = zip(img_dataset, img_labels)
    z_list = list(z)
    random.shuffle(z_list)
    img_dataset_tuple, img_labels_tuple = zip(*z_list)
    img_dataset_in = np.array(img_dataset_tuple)
    img_labels_in = np.array(img_labels_tuple)

    return img_dataset_in, img_labels_in

def get_slice(img_in_class, split, iteration, is_last=False):
    return math.ceil((img_in_class * split) / iteration) if is_last else math.floor((img_in_class * split) / iteration)


def get_keys(group):
    return re.split('/', group)

class h5_object():
    def __init__(self, folder_batch_size:int, get_key, get_ppm_arr, training_split=0.7):
        self.folder_batch_size = folder_batch_size
        self.nested_level = len(get_h5_path().split("/"))
        self.h5 = get_h5(get_h5_path())
        self.training_split = training_split
        self.get_key = get_key
        self.get_ppm_arr = get_ppm_arr

    
    def get_val_size(self):
        return 1 - self.training_split

    def generate_ppm_keys(self, start_val:int, end_val:int)->list:
        names = []
        for i in range(start_val, end_val):
            ppm_start = str(math.floor(i / self.folder_batch_size)).zfill(5)
            ppm_end = str(i % self.folder_batch_size).zfill(5)
            ppm = f"{ppm_start}_{ppm_end}.ppm"
            names.append(ppm)
        return names

    def append_to_list(self, ppm_names, keys, images, labels):
        for j in range(len(ppm_names)):
            arr = np.array(self.get_ppm_arr(self.h5, keys, ppm_names[j]))
            images.append(arr / 255.0)
            labels.append(int(keys[2]))


    def print_class_data(self):
        for i in range(len(group)):
            keys = get_keys(group[i])
            if len(keys) != self.nested_level:
                continue
            img_in_class = len(self.get_key(self.h5, keys))
            print(f"{i} & {img_in_class} & {math.floor(img_in_class * self.training_split)} & {math.ceil(img_in_class * h5_object.get_val_size(self))} /")



    def lazyload_h5(self, current_iteration, max_iteration, shuffle=True):
        is_last = current_iteration == max_iteration - 1

        train_set = []
        train_label = []

        val_set = []
        val_label = []

        print(f"groups: {len(group)}")
        print(f"data: {len(data)}")
        print(f"{current_iteration} / {max_iteration}")

        for i in range(len(group)):
            keys = get_keys(group[i])

            if len(keys) != self.nested_level:
                continue
            img_in_class = len(self.get_key(self.h5, keys))

            train_slice = get_slice(img_in_class, self.training_split, max_iteration)
            val_slice = get_slice(img_in_class, 1 - self.training_split, max_iteration, is_last)
            
            start_val = train_slice * current_iteration
            end_val = train_slice * current_iteration + train_slice if not is_last else math.ceil(img_in_class * self.training_split)
            ppm_names = h5_object.generate_ppm_keys(self, start_val, end_val)

            h5_object.append_to_list(self, ppm_names, keys, train_set, train_label)

            start_val = math.ceil(img_in_class * self.training_split) + (val_slice * current_iteration) if not is_last else math.ceil(img_in_class * self.training_split) + (max_iteration - 1) * (math.floor((img_in_class * (h5_object.get_val_size(self))) / max_iteration))
            end_val = math.ceil(img_in_class * self.training_split) + (val_slice * current_iteration + val_slice) if not is_last else img_in_class
            ppm_names = h5_object.generate_ppm_keys(self, start_val, end_val)

            h5_object.append_to_list(self, ppm_names, keys, val_set, val_label)

        if shuffle:
            train_set, train_label = Shuffle(train_set, train_label)
            val_set, val_label = Shuffle(val_set, val_label)


        return train_set, train_label, val_set, val_label