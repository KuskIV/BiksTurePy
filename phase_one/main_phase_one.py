from find_ideal_model import get_processed_models, train_and_eval_models_for_size, get_belgium_model
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
from tqdm import trange
import os
import sys,inspect



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

def find_ideal_model(h5_obj:object, model_object_list, epochs=10, lazy_split=10)->None:
    """finds the ideal model

    Args:
        h5_obj (object): h5 object
    """

    train_images = []
    test_images = []
    for j in range(lazy_split):
        
        # generate models
        train_images, train_labels, test_images, test_labels = h5_obj.shuffle_and_lazyload(j, lazy_split)
        
        print(f"Images in train_set: {len(train_images)} ({len(train_images) == len(train_labels)}), Images in val_set: {len(test_images)} ({len(test_images) == len(test_labels)})")
        print(f"This version will split the dataset in {lazy_split} sizes.")

        # train models
        for i in range(len(model_object_list)):
            print(f"Training model {i + 1} / {len(model_object_list) } for epoch {j + 1} / {lazy_split}")
            train_and_eval_models_for_size(model_object_list[i].model, model_object_list[i].size, i, train_images, train_labels, test_images, test_labels, epochs)
    
    for models in model_object_list:
        store_model(models.model, models.path)


    # large_model_path, medium_model_path, small_model_path, belgium_model_path = get_test_model_paths()

    # #store_model(models[0], large_model_path)
    # #store_model(models[1], medium_model_path)
    # # store_model(models[0], small_model_path) # SHOULD BE INDEX 2
    # store_model(models[0],belgium_model_path)


if __name__ == "__main__":
    lazy_split = 1
    dataset_split = 0.8

    image_sizes = [(82, 82)]
    # image_sizes = [(32, 32)]

    test_path = get_h5_test()
    train_path = get_h5_train()


    large_model_path, medium_model_path, small_model_path, belgium_model_path = get_test_model_paths()
    
    #large_model = tf.keras.models.load_model(large_model_path)
    #medium_model = tf.keras.models.load_model(medium_model_path)
    # small_model = tf.keras.models.load_model(small_model_path)
    belgium_model = tf.keras.models.load_model(belgium_model_path)

    # loaded_models = [large_model, medium_model, small_model]
    loaded_models = [belgium_model]
    

    #image_dataset = auto_reshape_images(image_sizes[0], image_dataset)



    # models = [get_processed_models(input_layer_size=h5_train.class_in_h5)[1]] # SHOULD NOT BE A LIST

    # find_ideal_model(h5_train, image_sizes, models, epochs=50, lazy_split=lazy_split)



    path = "/home/biks/Desktop"
    # test_path = "/imagesForMads"

    # class_names = open(get_paths("txt_file"), 'r').readlines()

    # models = [get_processed_models()[2]] # SHOULD NOT BE A LIST

    models = [get_belgium_model(input_layer_size=h5_train.class_in_h5)]

    image_dataset, lable_dataset, _, _ = h5_test.shuffle_and_lazyload(0, 1)

    

    # big_boi_list = [['Epochs', 'Resolution', 'Class', 'Class_Acuracy', 'Total_in_Class']]

    
    #acc_dist_for_images(h5_obj, [large_model, medium_model, small_model], [(200, 200), (128, 128), (32, 32)], 20)
    #acc_dist_for_images(h5_obj, [large_model_path, medium_model_path, small_model_path], 10)
    
    # This was a table generator for roni
    # h5_obj.print_class_data()