from find_ideal_model import get_processed_models, train_and_eval_models_for_size
import numpy as np
from PIL import Image
import math
import h5py
import tensorflow as tf

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from Dataset.load_h5 import h5_object
from Models.create_model import store_model
from Models.test_model import partial_accumilate_distribution, print_accumilate_distribution
from global_paths import get_test_model_paths
from general_image_func import auto_reshape_images

lazy_split = 10
dataset_split = 0.7
folder_batch_size = 30

def acc_dist_for_images(h5_obj:object, models:list, sizes:list, lazy_split)->None:
    accArr = np.zeros((3, 43, 2))

    for k in range(lazy_split):
        train_images, train_labels, _, _ = h5_obj.shuffle_and_lazyload(k, lazy_split)
        for i in range(len(models)):
            train_images = auto_reshape_images(sizes[i], train_images, smart_resize=True)
            arr = partial_accumilate_distribution(train_images, train_labels, sizes[i], model=models[i])
            for j in range(len(arr)):
                accArr[i][j][0] += arr[j][0]
                accArr[i][j][1] += arr[j][1]
        #break
    
    for i in range(len(models)):
        print_accumilate_distribution(accArr[i], size=sizes[i])


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
    h5_obj = h5_object(folder_batch_size, training_split=dataset_split)
    #find_ideal_model(h5_obj)
    
    large_model_path, medium_model_path, small_model_path = get_test_model_paths()
    
    large_model = tf.keras.models.load_model(large_model_path)
    medium_model = tf.keras.models.load_model(medium_model_path)
    small_model = tf.keras.models.load_model(small_model_path)
    
    acc_dist_for_images(h5_obj, [large_model, medium_model, small_model], [(200, 200), (128, 128), (32, 32)], 20)
    # acc_dist_for_images(h5_obj, [large_model_path, medium_model_path, small_model_path], 10)
    
    # # This was a table generator for roni
    # h5_obj.print_class_data()