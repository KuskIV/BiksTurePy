from find_ideal_model import get_processed_models, train_and_eval_models_for_size
import numpy as np
from PIL import Image
import math
import h5py
from matplotlib import pyplot as plt
import tensorflow as tf
import os, logging
import sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from Dataset.load_h5 import h5_object
from Models.create_model import store_model
from Models.test_model import partial_accumilate_distribution, print_accumilate_distribution, make_prediction
from global_paths import get_test_model_paths, get_paths
from general_image_func import auto_reshape_images, changeImageSize, convert_numpy_image_to_image

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
    
    for i in range(len(models)):
        print_accumilate_distribution(accArr[i], size=sizes[i])


def find_ideal_model(h5_obj:object, epochs=10)->None:
    """finds the ideal model

    Args:
        h5_obj (object): h5 object
    """
    #image_sizes = [(200, 200), (128, 128), (32, 32)]
    image_sizes = [(32, 32)]

    models = [get_processed_models()[2]] # SHOULD NOT BE A LIST

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
            train_and_eval_models_for_size(models, model_and_size[i][1], model_and_size[i][0], i, train_images, train_labels, test_images, test_labels, epochs)
    large_model_path, medium_model_path, small_model_path = get_test_model_paths()

    #store_model(models[0], large_model_path)
    #store_model(models[1], medium_model_path)
    store_model(models[0], small_model_path) # SHOULD BE INDEX 2

def get_rid_of_a(arr:np.array):
    w, h = arr.size
    if arr.getdata().mode == 'RGBA':
        arr = arr.convert('RGB')
    nparray = np.array(arr.getdata())
    reshaped = nparray.reshape((w, h, 3))
    return reshaped.astype(np.uint8)

def loadImags(folder, lable):
    loaded_img = []
    lable_names = []
    with os.scandir(folder) as imgs:
        for ppm_path in imgs:
            if ppm_path.name.endswith(".jpg") or ppm_path.name.endswith('.jpeg') or ppm_path.name.endswith('.ppm'):
                lable_names.append(lable)
                #print(ppm_path.path)
                # loaded_img.append(get_rid_of_a(Image.open(ppm_path.path)))
                loaded_img.append(np.asanyarray(Image.open(ppm_path.path)) / 255.0)
    return lable_names, auto_reshape_images((32, 32), loaded_img)  

def load_X_images(path):
    subfolders = [ f.path for f in os.scandir(path) if f.is_dir() ]
    newImgs = []
    lable_names = []
    for folder in subfolders:
        lable = folder.split('/')[-1]
        returned_lables, imgs = loadImags(folder, lable)
        newImgs.extend(imgs)
        lable_names.extend(returned_lables)
    return newImgs, lable_names

if __name__ == "__main__":
    h5_obj = h5_object(folder_batch_size, training_split=dataset_split)
    
    path = "/home/biks/Desktop"
    test_path = "/imagesForMads"

    class_names = open(get_paths("txt_file"), 'r').readlines()

    image_dataset, lable_dataset = load_X_images(path + test_path)
    label_dict = {}
    
    for i in range(3, 15):
        print(f"Run {i} / {15}")
        find_ideal_model(h5_obj, epochs=i)
        
        large_model_path, medium_model_path, small_model_path = get_test_model_paths()
        
        #large_model = tf.keras.models.load_model(large_model_path)
        #medium_model = tf.keras.models.load_model(medium_model_path)
        small_model = tf.keras.models.load_model(small_model_path)
        
        
        for lable in lable_dataset:
            label_dict[lable] = [0,0]
        
        right = 0
        wrong = 0 
        
        for i in range(len(image_dataset)):
            prediction = make_prediction(small_model, image_dataset[i].copy())
            predicted_label = np.argmax(prediction)
            softmaxed = tf.keras.activations.softmax(prediction)
            img = convert_numpy_image_to_image(image_dataset[i])

            if int(predicted_label) == int(lable_dataset[i]):
                right += 1
                label_dict[lable_dataset[i]][1] = label_dict[lable_dataset[i]][1] +1
                #img.save(f"{path}/right/{i}_{predicted_label}_{int(100*np.max(softmaxed))}_{ephocs}.jpg")
            else:
                wrong += 1
                label_dict[lable_dataset[i]][0] = label_dict[lable_dataset[i]][0] +1
                #img.save(f"{path}/wrong/{i}_{lable_dataset[i]}_{predicted_label}_{int(100*np.max(softmaxed))}_{ephocs}.jpg")

        print(f"Right: {right}, wrong: {wrong}, percent correct: {(right / (wrong + right)) * 100}")
        for key in label_dict.keys():
            class_name = str(key).zfill(2)
            right_name = str(label_dict[key][1]).rjust(4, ' ')
            wrong_name = str(label_dict[key][0]).rjust(4, ' ')
            percent = (label_dict[key][1]/(label_dict[key][0]+label_dict[key][1])*100)

            print(f"class: {class_name} | right: {right_name} | wrong: {wrong_name} | procent: {round(percent, 2)}")


    
    #acc_dist_for_images(h5_obj, [large_model, medium_model, small_model], [(200, 200), (128, 128), (32, 32)], 20)
    # # acc_dist_for_images(h5_obj, [large_model_path, medium_model_path, small_model_path], 10)
    
    # # This was a table generator for roni
    # h5_obj.print_class_data()