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

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from general_image_func import Shuffle

data = []
group = []

def sort_groups(name, obj):
    if isinstance(obj, h5py.Dataset):
        data.append(name)
    elif isinstance(obj, h5py.Group):
        group.append(name)


def read_h5(h5_path): # Not yet implemented correctly
    h5 = h5py.File(h5_path, 'r')
    print(h5['Dataset']['images'])
    #print(test)
            
    h5.visititems(sort_groups)

    print(f"groups: {len(group)}")
    print(f"data: {len(data)}")

    #for j in data:
    #    kk = np.array(h5[j])
    #    img = Image.open(io.BytesIO(kk))
    #    print('image size: ', img.size)

def get_h5(h5_path):
    if not path.exists(h5_path):
        print(f"The path for the h5 file does not exist ({h5_path}). The program has exited.")
        sys.exit()
    else:
        h5 = h5py.File(h5_path, 'r')
        h5.visititems(sort_groups)
        return h5

def get_min_max(imgCount, training_split, split, current_split, lastIndex):
    percent = imgCount * training_split
    minVal = (percent / split) * (current_split)
    maxVal = minVal + (percent / split)

    return math.floor(minVal), math.floor(maxVal) if not lastIndex else math.ceil(maxVal)

def append_to_lists(index, img_per_class, training_split, split, current_split, srcImg, srcLabel, outImg, outLabel, lastIndex=False):
    minVal, maxVal = get_min_max(img_per_class, training_split, split, current_split, lastIndex)

    minVal += index
    maxVal += index

    #print(f"{img_per_class}, min: {minVal}, max: {maxVal}")

    for j in range(minVal, maxVal):
        outImg.append(srcImg[j])
        outLabel.append(srcLabel[j])
    



def lazy_split(h5, images_per_class, split, current_split, lastIndex, training_split:float=.7, shuffle:bool=True)->tuple:
    minIndex = 0
    maxIndex = 0

    train_set = []
    train_label = []

    val_set = []
    val_label = []




    # for i in range(len(images_per_class)):
    #     if i != 0:
    #         minIndex += images_per_class[i - 1]
    #     maxIndex += math.floor(images_per_class[i - 1 if i > 0 else 0] * training_split)

    #     append_to_lists(minIndex, images_per_class[i], training_split, split, current_split, img_dataset, img_labels, train_set, train_label, lastIndex)
    #     append_to_lists(maxIndex, images_per_class[i], 1 - training_split, split, current_split, img_dataset, img_labels, val_set, val_label)

    # if shuffle:
    #     val_set, val_label = Shuffle(val_set, val_label)
    #     train_set, train_label = Shuffle(train_set, train_label)

    return train_set, train_label, val_set, val_set

def generate_ppm_keys(start_val, end_val):
    folder_batch_size = 30
    for i in range(start_val, end_val):
        ppm_start = str(math.floor(i / folder_batch_size)).zfill(5)
        ppm_end = str(i % folder_batch_size).zfill(5)
        print(f"{ppm_start}_{ppm_end}.ppm")


def get_slice(img_in_class, split, iteration, is_last=False):
    return math.ceil((img_in_class * split) / iteration) if is_last else math.floor((img_in_class * split) / iteration)

def lazyload_h5(h5, current_iteration, max_iteration, training_split:float=.7):
    is_last = current_iteration == max_iteration - 1
    nested_level = 3
    folder_batch_size = 30

    train_set = []
    train_label = []

    val_set = []
    val_label = []



    print(f"groups: {len(group)}")
    print(f"data: {len(data)}")

    for i in range(len(group)):
        keys = re.split('/', group[i]) # this is kinda fucked

        if len(keys) != nested_level:
            continue

        #print(h5[keys[0]][keys[1]][keys[2]])
        img_in_class = len(h5[keys[0]][keys[1]][keys[2]])
        #print(img_in_class)
        print(f"In class: {img_in_class}, train: {img_in_class * training_split}, val: {img_in_class * (1 - training_split)}")
        if img_in_class == 210:      
            train_slice = get_slice(img_in_class, training_split, max_iteration) # '10' should be replaced by 'max_split'
            val_slice = get_slice(img_in_class, 1 - training_split, max_iteration, is_last)
            

            print(f"Start: {train_slice * current_iteration}, End: {train_slice * current_iteration + train_slice }")
            print(f"Start: {math.ceil(img_in_class * training_split) + (val_slice * current_iteration)}, End: {math.ceil(img_in_class * training_split) + (val_slice * current_iteration + val_slice)}")
            print("---")
            
            start_val = train_slice * current_iteration
            end_val = train_slice * current_iteration + train_slice
            generate_ppm_keys(start_val, end_val)

            print("---")
            start_val = math.ceil(img_in_class * training_split) + (val_slice * current_iteration)
            end_val = math.ceil(img_in_class * training_split) + (val_slice * current_iteration + val_slice)
            generate_ppm_keys(start_val, end_val)
            

            #print(train_slice * current_iteration, " - ", train_slice, " - ", val_slice * current_iteration, " - ", val_slice)
            break

        
    return [], [], [], []