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

def find_ideal_model():
    img_dataset = [] # list of all images in reshaped numpy array
    img_labels = [] # labels for all images in correct order
    images_per_class = [] # list, where each entry represents the number of ppm images for that classification class
    class_names = [] # classification text for labels

    class_names = get_class_names()

    image_sizes = [(32, 32), (128, 128), (200, 200)]

    #img_dataset, img_labels, images_per_class = get_data(fixed_size = (32, 32), padded_images = False, smart_resize = True)
    
    # Training and test split, 70 and 30%
    lazy_split = 10
    dataset_split = 0.7
    folder_batch_size = 30

    h5_obj = h5_object(folder_batch_size, dataset_split)
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
    # stor each model
    # if save_model:
    #     #store model in saved_models with name as img_shape X model design
    #     filename = 'adj' + str(size[0])
    #     model_id = models.index(model)
    #     if model_id == 0:
    #         filename += "default"
    #     elif model_id == 1:
    #         filename += "medium"
    #     else:
    #         filename += "large"
    store_model(models[0], "default32")
    store_model(models[1], "medium128")
    store_model(models[2], "large200")

if __name__ == "__main__":
    find_ideal_model()