from find_ideal_model import train_and_eval_models_for_size, get_belgium_model
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
from tqdm import trange
import os
import sys,inspect
from experiment_one import run_experiment_one

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from plot.write_csv_file import cvs_object, plot
from Dataset.load_h5 import h5_object
from Models.create_model import store_model
from Models.test_model import partial_accumilate_distribution, print_accumilate_distribution, make_prediction
from global_paths import get_test_model_paths, get_paths, get_h5_test, get_h5_train
from general_image_func import auto_reshape_images, convert_numpy_image_to_image, load_images_from_folders

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
    
    for i in range(len(models)):
        print_accumilate_distribution(accArr[i], size=sizes[i])

if __name__ == "__main__":
    lazy_split = 1

    test_path = get_h5_test()
    train_path = get_h5_train()

    run_experiment_one(lazy_split, train_path, test_path, epochs=(1,4))

    
    # path = "/home/biks/Desktop"
    # test_path = "/imagesForMads"

    # class_names = open(get_paths("txt_file"), 'r').readlines()

    