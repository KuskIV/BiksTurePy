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
from matplotlib import pyplot as plt

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import Dataset.os_h5_constructor as osc
from global_paths import get_h5_path

group = []
data = []

def reset_read_h5():
    global group, data

    group = []
    data = []

def sort_groups(name:int, obj:object)->None:
    """Used to sort objects in a H5PY file into grous and data

    Args:
        name (int): The name of the class
        obj (object): the object we are looking at
    """
    if isinstance(obj, h5py.Dataset):
        data.append(name)
    elif isinstance(obj, h5py.Group):
        group.append(name)


def get_h5(h5_path:str)->h5py._hl.files.File:
    """Given a path, a H5PY object is returned if it exists

    Args:
        h5_path (str): The path where the H5PY is located

    Returns:
        h5py._hl.files.File: The opend H5PY
    """
    if not path.exists(h5_path):
        print(f"The path for the h5 file does not exist ({h5_path}). The program has exited.")
        sys.exit()
    else:
        h5 = h5py.File(h5_path, 'r')
        h5.visititems(sort_groups)
        return h5

def get_h5_and_lists(h5_path:str)->h5py._hl.files.File:
    """Given a path, a H5PY object is returned if it exists

    Args:
        h5_path (str): The path where the H5PY is located

    Returns:
        h5py._hl.files.File: The opend H5PY
    """
    if not path.exists(h5_path):
        print(f"The path for the h5 file does not exist ({h5_path}). The program has exited.")
        sys.exit()
    else:
        h5 = h5py.File(h5_path, 'r')
        h5.visititems(sort_groups)
        return h5, group, data


## DONT DELETE THESE THREE LINES OF CODE!
#img_as_arr = np.array(self.h5[keys[0]][keys[1]][ppm_names[i]])
#img = Image.fromarray(img_as_arr.astype('uint8'), 'RGB')
#img.show()

def Shuffle(img_dataset:int, img_labels:int)->tuple:
    """Given two lists, they are zipped together, are shuffled and returned

    Args:
        img_dataset (int): A list of arrays, representing the images
        img_labels (int): A list of lables for each of the images

    Returns:
        tuple: A tuple consisting of the images and lables after they are shuffled
    """
    img_dataset_in = img_dataset
    img_labels_in = img_labels

    z = zip(img_dataset, img_labels)
    z_list = list(z)
    random.shuffle(z_list)
    img_dataset_tuple, img_labels_tuple = zip(*z_list)
    img_dataset_in = np.array(img_dataset_tuple)
    img_labels_in = np.array(img_labels_tuple)

    return img_dataset_in, img_labels_in

class h5_object():
    
    # def generate_ppm_keys(self, start_val:int, end_val:int)->list: # Not used anymore
    #     """Generates the names of the ppm images based on a start and end value

    #     Args:
    #         start_val (int): The first ppm image name to be generated
    #         end_val (int): the last ppm image name to be generated

    #     Returns:
    #         list: a list of all the ppm image names
    #     """
    #     names = []
    #     for i in range(start_val, end_val):
    #         ppm_start = str(math.floor(i / self.folder_batch_size)).zfill(5)
    #         ppm_end = str(i % self.folder_batch_size).zfill(5)
    #         ppm = f"{ppm_start}_{ppm_end}.ppm"
    #         self.img_in_h5 += 1
    #         names.append(ppm)
    #     return names

    def ppm_keys_to_list(self)->None:
        """Generates names for all ppm images based on how many ppm images there are in each folder
        """

        for i in range(len(self.group)):
            keys = self.get_keys(self.group[i])
            if len(keys) > self.nested_level:
                self.ppm_names.append([])
                self.class_in_h5 += 1
                for j in self.get_key(self.h5, keys):
                    self.ppm_names[-1].append(j)
                self.images_in_classes[keys[-1]] = len(self.ppm_names[-1]) 
                # print(f"class {keys[-1]}, len {len(self.ppm_names[-1])}")
                random.shuffle(self.ppm_names[-1])

    def __init__(self, h5_path:str, training_split=0.7, os_constructor=osc.get_os_constructor):
        self.h5, self.group, self.data = get_h5_and_lists(h5_path)
        reset_read_h5()

        self.training_split = training_split
        
        os_tuple = os_constructor()

        
        self.os = os_tuple[0]
        self.key_to_string = os_tuple[1]
        self.get_ppm_arr = os_tuple[2]
        self.get_key = os_tuple[3]
        self.get_keys = os_tuple[4]

        self.error_index = 2 if self.os == 'windows' else 3                                         #TODO: FIX, NOT HARDCODE
        self.nested_level = len(re.split('/', get_h5_path())) - (1 if self.os == 'windows' else 0)  #TODO: FIX NOT HARDCODE, BASED ON PATH

        self.images_in_classes = {}
        self.ppm_names = []
        self.img_in_h5 = 0
        self.class_in_h5 = 0
        
        
        self.ppm_keys_to_list()


    def get_ppm_img_index(self, index:int)->int:
        """Returns the index of class currently in, subtracting the unused ones

        Args:
            index (int): the current index

        Returns:
            [type]: the index when the error index is subtracted
        """
        return index - self.error_index
    


    def get_val_size(self)->float:
        """Calculates and returns the valuation size, by subtracting the training split

        Returns:
            float: the valuation size
        """
        return 1 - self.training_split

    def append_to_list(self, ppm_names:list, keys:list, images:list, labels:list):
        """Given a list of ppm names, they are translated into their corresponding image, and added to a list, alongisde its lable

        Args:
            ppm_names (list): the list of names of ppm images
            keys (list): The key for the list, containing the class
            images (list): the list of image to add to
            labels (list): the list of lables
        """
        for j in range(len(ppm_names)):
            arr = np.array(self.get_ppm_arr(self.h5, keys, ppm_names[j]))
            images.append(arr)
            labels.append(int(keys[-1]) if self.os == 'linux' else int(keys[-1].split('\\')[1]))

    def print_class_data(self)->None: #TODO: data to csv file
        """A table generator method for latex, which prints out the amount of images in each class
        """
        for i in range(len(self.group)):
            keys = self.get_keys(self.group[i])
            if len(keys) != self.nested_level:
                continue
            img_in_class = len(self.get_key(self.h5, keys))
            print(f"{i-2} & {img_in_class} & {math.floor(img_in_class * self.training_split)} & {math.ceil(img_in_class * h5_object.get_val_size(self))} /")

    def get_part_of_array(self, current_slize:int, max_slice:int, split:float, class_index:int, train_set:list, train_label:list, val_set:list, val_label:list, keys:list, i)->tuple:
        """Returns a part of the train_set, train_lable, val_set and val_label lists, based on how many slices the lists are split into and what slize we are currently on

        Args:
            current_slize (int): the current slize 
            max_slice (int): how many slizes the array should be split itno
            split (float): how the train and validation should be split
            class_index (int): the class of images currently to be split
            train_set (list): the list of train images
            train_label (list): the list of train labesl
            val_set (list): the list of validation images
            val_label (list): the list of validation lables
            keys (list): the key to the images from the h5 file
        """
        is_last = current_slize == max_slice - 1

        split_size = math.floor(len(self.ppm_names[class_index]) / max_slice)

        train_size = math.floor(split_size * split)
        val_size = split_size - train_size

        train_start = split_size * current_slize
        train_end = train_start + train_size
        
        val_start = train_end
        val_end = val_start + val_size if not is_last else len(self.ppm_names[class_index])

        # print(f"{str(self.group[i]).zfill(2)} - train: {str(train_end - train_start).rjust(5, ' ')} | {str(train_start).rjust(6, ' ')} - {str(train_end).ljust(6, ' ')} - val: {str(val_end - val_start).rjust(5, ' ')} | {str(val_start).rjust(6, ' ')} - {str(val_end).ljust(6, ' ')}")
        
        self.append_to_list(self.ppm_names[class_index][train_start:train_end], keys, train_set, train_label)
        self.append_to_list(self.ppm_names[class_index][val_start:val_end], keys, val_set, val_label)

        return train_set, train_label, val_set, val_label

    def shuffle_and_lazyload(self, current_iteration:int, max_iteration:int, shuffle=True)->tuple:
        """This method iterates through all the different classes of images, and returns a part of the images based on the current iteration and the max iteratino

        Args:
            current_iteration (int): the current iteration
            max_iteration (int): how many slizes the images should be split into
            shuffle (bool, optional): whether or not the output tuple should be shuffled. Defaults to True.

        Returns:
            tuple: a tuple consisting of a train and validation set each containing images and labes.
        """
        train_set = []
        train_label = []

        val_set = []
        val_label = []

        for i in range(len(self.group)):
            keys = self.get_keys(self.group[i])

            if len(keys) > self.nested_level:
                h5_object.get_part_of_array(self, current_iteration, max_iteration, self.training_split, self.get_ppm_img_index(i), train_set, train_label, val_set, val_label, keys, i) 

        print(f"{current_iteration}: train set: {len(train_set)}, train lables: {len(train_label)}, val set: {len(val_set)}, val labels: {len(val_label)}") 

        if shuffle:
            if len(train_set) > 0:
                train_set, train_label = Shuffle(train_set, train_label)
            if len(val_set) > 0:
                val_set, val_label = Shuffle(val_set, val_label)

        

        return train_set, train_label, val_set, val_label

