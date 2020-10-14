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



class h5_object():
    def __init__(self, folder_batch_size, training_split=0.7):
        self.folder_batch_size = folder_batch_size
        self.nested_level = 2 #len(get_h5_path().split("/"))
        self.h5 = get_h5(get_h5_path())
        self.training_split = training_split

    
    def get_val_size(self):
        return 1 - self.training_split

    def generate_ppm_keys(self, start_val, end_val):
        names = []
        #print(end_val - start_val)
        for i in range(start_val, end_val):
            ppm_start = str(math.floor(i / self.folder_batch_size)).zfill(5)
            ppm_end = str(i % self.folder_batch_size).zfill(5)
            ppm = f"{ppm_start}_{ppm_end}.ppm"
            names.append(ppm)
            #print(ppm)

            #img_as_arr = np.array(self.h5[a][b][ppm])
            #img = Image.fromarray(img_as_arr.astype('uint8'), 'RGB')
            #img.show()
        return names

    def append_to_list(self, ppm_names, keys, images, labels):
        for j in range(len(ppm_names)):
            arr = np.asarray(self.h5[keys[0]][keys[1]][ppm_names[j]])# Add keys[2] # TODO make get_key method
            images.append(arr.flatten())
            labels.append(keys[1]) # Should be key 2

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
            keys = re.split('/', group[i])
            #print(keys)
            if len(keys) != self.nested_level:
                continue
            img_in_class = len(self.h5[keys[0]][keys[1]]) # add: keys[2] # TODO make get_key method
            #print(keys)
            #print(img_in_class)

            train_slice = get_slice(img_in_class, self.training_split, max_iteration)
            val_slice = get_slice(img_in_class, 1 - self.training_split, max_iteration, is_last)
            
            # print(f"Start: {train_slice * current_iteration}, End: {train_slice * current_iteration + train_slice }")
            # print(f"Start: {math.ceil(img_in_class * self.training_split) + (val_slice * current_iteration)}, End: {math.ceil(img_in_class * self.training_split) + (val_slice * current_iteration + val_slice)}")
            # print("---")
            
            start_val = train_slice * current_iteration
            end_val = train_slice * current_iteration + train_slice if not is_last else math.ceil(img_in_class * self.training_split)
            ppm_names = h5_object.generate_ppm_keys(self, start_val, end_val)
            
            #img_as_arr = np.array(self.h5[keys[0]][keys[1]][ppm_names[0]])
            #img = Image.fromarray(img_as_arr.astype('uint8'), 'RGB')
            #img.show()
            #break

            h5_object.append_to_list(self, ppm_names, keys, train_set, train_label)

            start_val = math.ceil(img_in_class * self.training_split) + (val_slice * current_iteration) if not is_last else math.ceil(img_in_class * self.training_split) + (max_iteration - 1) * (math.floor((img_in_class * (h5_object.get_val_size(self))) / max_iteration))
            end_val = math.ceil(img_in_class * self.training_split) + (val_slice * current_iteration + val_slice) if not is_last else img_in_class
            ppm_names = h5_object.generate_ppm_keys(self, start_val, end_val)

            h5_object.append_to_list(self, ppm_names, keys, val_set, val_label)

        if shuffle:
            train_set, train_label = Shuffle(train_set, train_label)
            val_set, val_label = Shuffle(val_set, val_label)


        return train_label, train_label, val_set, val_label