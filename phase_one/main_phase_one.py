from find_ideal_model import get_processed_models, train_and_eval_models_for_size
import numpy as np
from PIL import Image
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from general_image_func import get_class_names # This is not an error
from Dataset.load_h5 import h5_object
from global_paths import get_h5_path
from Models.create_model import store_model

def get_ppm_arr(h5, keys, ppm_name):
    return h5[keys[0]][keys[1]][keys[2]][ppm_name]

def get_key(h5, keys):
    return h5[keys[0]][keys[1]][keys[2]]

def find_ideal_model():
    class_names = get_class_names()

    image_sizes = [(32, 32), (128, 128), (200, 200)]

    # Training and test split, 70 and 30%
    lazy_split = 10
    dataset_split = 0.7
    folder_batch_size = 30

    h5_obj = h5_object(folder_batch_size, get_key, get_ppm_arr, dataset_split)
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

    store_model(models[0], "default32")
    store_model(models[1], "medium128")
    store_model(models[2], "large200")

if __name__ == "__main__":
    find_ideal_model()