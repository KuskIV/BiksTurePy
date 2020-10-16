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

from global_paths import get_h5_path

data = []
group = []

def sort_groups(name:int, obj:object)->None:
    if isinstance(obj, h5py.Dataset):
        data.append(name)
    elif isinstance(obj, h5py.Group):
        group.append(name)

def get_h5(h5_path:str)->h5py._hl.files.File:
    if not path.exists(h5_path):
        print(f"The path for the h5 file does not exist ({h5_path}). The program has exited.")
        sys.exit()
    else:
        h5 = h5py.File(h5_path, 'r')
        h5.visititems(sort_groups)
        return h5


## DONT DELETE THESE THREE LINES OF CODE!
#img_as_arr = np.array(self.h5[keys[0]][keys[1]][ppm_names[i]])
#img = Image.fromarray(img_as_arr.astype('uint8'), 'RGB')
#img.show()

def Shuffle(img_dataset:int, img_labels:int)->tuple:
    img_dataset_in = img_dataset
    img_labels_in = img_labels

    z = zip(img_dataset, img_labels)
    z_list = list(z)
    random.shuffle(z_list)
    img_dataset_tuple, img_labels_tuple = zip(*z_list)
    img_dataset_in = np.array(img_dataset_tuple)
    img_labels_in = np.array(img_labels_tuple)

    return img_dataset_in, img_labels_in

def get_slice(img_in_class:int, split:float, iteration:int, is_last=False)->int:
    return math.ceil((img_in_class * split) / iteration) if is_last else math.floor((img_in_class * split) / iteration)


def get_keys(group)->str:
    return re.split('/', group)

class h5_object():
    
    def generate_ppm_keys(self, start_val:int, end_val:int)->list:
        names = []
        for i in range(start_val, end_val):
            ppm_start = str(math.floor(i / self.folder_batch_size)).zfill(5)
            ppm_end = str(i % self.folder_batch_size).zfill(5)
            ppm = f"{ppm_start}_{ppm_end}.ppm"
            names.append(ppm)
        return names

    def __init__(self, folder_batch_size:int, get_key, get_ppm_arr, train_set_start_end, val_set_start_end, training_split=0.7):
        self.folder_batch_size = folder_batch_size
        self.nested_level = len(get_h5_path().split("/"))
        self.h5 = get_h5(get_h5_path())
        self.training_split = training_split
        
        self.get_key = get_key
        self.get_ppm_arr = get_ppm_arr
        self.train_set_start_end = train_set_start_end
        self.val_set_start_end = val_set_start_end

        self.ppm_names = []
        for i in range(len(group)):
            keys = get_keys(group[i])
            if len(keys) == self.nested_level:
                img_in_class = len(self.get_key(self.h5, keys))
                self.ppm_names.append(self.generate_ppm_keys(0, img_in_class))
                random.shuffle(self.ppm_names[-1])
                print(self.ppm_names[-1])


    def get_val_size(self)->float:
        return 1 - self.training_split



    def append_to_list(self, ppm_names:list, keys:list, images:list, labels:list):
        for j in range(len(ppm_names)):
            arr = np.array(self.get_ppm_arr(self.h5, keys, ppm_names[j]))
            images.append(arr / 255.0)
            labels.append(int(keys[2]))


    def print_class_data(self)->None:
        for i in range(len(group)):
            keys = get_keys(group[i])
            if len(keys) != self.nested_level:
                continue
            img_in_class = len(self.get_key(self.h5, keys))
            print(f"{i-2} & {img_in_class} & {math.floor(img_in_class * self.training_split)} & {math.ceil(img_in_class * h5_object.get_val_size(self))} /")

    # def get_part_of_array(self, current_slize, max_slice, split, class_index, train_set, train_label, val_set, val_label):
    #     is_last = current_slize == max_slice - 1
    #     split_size = math.floor(len(self.ppm_names) / max_slice)

    #     train_size = math.floor(split_size * split)
    #     val_size = split_size - train_size

    #     train_start = split_size * current_slize
    #     train_end = train_start + train_size
    #     val_start = train_end
    #     val_end = val_start + val_size if not is_last else len(self.ppm_names)

    #     for i in range(train_start, train_end):
    #         train_set.append(self.ppm_names[i]) # this should be an array, now it is the name
    #         train_label.append(class_index)
        
    #     for i in range(val_start, val_end):
    #         val_set.append(self.ppm_names[i]) # this should be an array, now it is the name
    #         val_label.append(class_index)

    #     return train_set, train_label, val_set, val_label

    # def shuffle_and_lazyload(self, current_iteration:int, max_iteration:int, shuffle=True)->tuple:
    #     train_set = []
    #     train_label = []

    #     val_set = []
    #     val_label = []

    #     for i in range(len(group)):
    #         keys = get_keys(group[i])

    #         if len(keys) == self.nested_level:
    #             img_in_class = len(self.get_key(self.h5, keys))
    #             h5_object.get_part_of_array(self, current_iteration, max_iteration, self.training_split, i, train_set, train_label, val_set, val_label) 
    #             # in line above, i should be the name of the class (folder name, int)


    #     if shuffle:
    #         train_set, train_label = Shuffle(train_set, train_label)
    #         val_set, val_label = Shuffle(val_set, val_label)

    #     return train_set, train_label, val_set, val_label

    def lazyload_h5(self, current_iteration:int, max_iteration:int, shuffle=True)->tuple:
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

            if len(keys) == self.nested_level:
                img_in_class = len(self.get_key(self.h5, keys))

                train_slice = get_slice(img_in_class, self.training_split, max_iteration)
                val_slice = get_slice(img_in_class, h5_object.get_val_size(self), max_iteration, is_last)
                
                start_val, end_val = self.train_set_start_end(self, train_slice, current_iteration, is_last, img_in_class)
                ppm_names = h5_object.generate_ppm_keys(self, start_val, end_val)

                h5_object.append_to_list(self, ppm_names, keys, train_set, train_label)
                
                start_val, end_val = self.val_set_start_end(self, train_slice, val_slice, current_iteration, is_last, img_in_class, max_iteration)
                ppm_names = h5_object.generate_ppm_keys(self, start_val, end_val)

                h5_object.append_to_list(self, ppm_names, keys, val_set, val_label)

        if shuffle:
            train_set, train_label = Shuffle(train_set, train_label)
            val_set, val_label = Shuffle(val_set, val_label)

        return train_set, train_label, val_set, val_label