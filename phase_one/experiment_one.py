import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tqdm import trange
import sys, os
import csv
import os.path
from os import path

from find_ideal_model import train_and_eval_models_for_size, get_model_object_list

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from Dataset.load_h5 import h5_object
from Models.create_model import store_model
from global_paths import get_test_model_paths, get_paths, get_h5_test, get_h5_train
from plot.write_csv_file import cvs_object, plot
from general_image_func import auto_reshape_images, convert_numpy_image_to_image
from Models.test_model import make_prediction
from plot.sum_csv import sum_csv

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
            train_and_eval_models_for_size(model_object_list[i].img_shape, model_object_list[i].model, i, train_images, train_labels, test_images, test_labels, epochs)
    
    for models in model_object_list:
        store_model(models.model, models.path)

def run_experiment_one(lazy_split, train_h5_path, test_h5_path, epochs=(3,4), dataset_split=0.7):
    label_dict = {}

    h5_train = h5_object(train_h5_path, training_split=dataset_split)
    h5_test = h5_object(test_h5_path, training_split=1)

    if h5_train.class_in_h5 != h5_test.class_in_h5:
        print(f"The input train and test set does not have matching classes {h5_train.class_in_h5} - {h5_test.class_in_h5}")
        sys.exit()

    model_object_list = get_model_object_list(h5_train.class_in_h5)

    image_dataset, lable_dataset, _, _ = h5_test.shuffle_and_lazyload(0, 1)

    
    
    for e in range(epochs[0], epochs[1]):
        print(f"\n----------------\nRun {e} / {epochs[1]}\n----------------\n")
        
        find_ideal_model(h5_train, model_object_list, lazy_split=lazy_split, epochs=e)
        
        print(f"\n------------------------\nTraining done. Now evaluation will be made, using {e} epochs.\n\n")

        iterate_trough_models(model_object_list, lable_dataset, label_dict, e, image_dataset)

    save_plot(model_object_list)
    sum_plot(model_object_list)
    sum_class_accuracy(model_object_list)

def sum_class_accuracy(model_object_list):
    model_class_accuracy = {}

    for model_object in model_object_list:
        model_class_accuracy[model_object.get_csv_name()] = {}

        with open(model_object.get_csv_path(), 'r') as csvfile:
                if not path.exists(model_object.get_csv_path()):
                    print(f"\nThe following path does not exist: {model_object.get_csv_path()}\nCode: plot.write_csv_file.py")
                    sys.exit()
                
                plots = csv.reader(csvfile, delimiter=',')

                next(plots)
                for row in plots:
                    if not row[0] in model_class_accuracy[model_object.get_csv_name()]: 
                        model_class_accuracy[model_object.get_csv_name()][row[0]] = {}
                    model_class_accuracy[model_object.get_csv_name()][row[0]][row[2]] = row[3]
    
    data_list = convert_dict_to_list(model_class_accuracy)
    save_data_obj = cvs_object(f"{get_paths('phase_one_csv')}/class_accuracy.csv")
    save_data_obj.write(data_list)

    return model_class_accuracy

def convert_dict_to_list(model_class_accuracy):
    data_list = [['Class']]

    for key, value in model_class_accuracy.items():
        for key2, value2 in model_class_accuracy[key].items():
            data_list[0].append(f"{key}_{key2}")
            
            i = 0
            while str(i) in value2:
                if len(data_list) == i + 1:
                    data_list.append([i])
                data_list[i + 1].append(model_class_accuracy[key][key2][str(i)])
                i += 1

    return data_list  

def sum_plot(model_object_list):
    csv_object_list =  []
    for model_object in model_object_list:
        obj = cvs_object(model_object.get_csv_path(), label=model_object.get_size())
        data = sum_csv(obj)
        obj.write(data, model_object.get_summed_csv_path(), overwrite_path=True)
        csv_object_list.append(obj)
    plot(csv_object_list)

def save_plot(model_object_list):
    for model_object in model_object_list:
        cvs_obj = cvs_object(model_object.get_csv_path())
        cvs_obj.write(model_object.csv_data)

def iniitalize_dict(lable_dataset):
    label_dict = {}
    for lable in lable_dataset:
            label_dict[lable] = [0,0]
    return label_dict


def iterate_trough_models(model_object_list,lable_dataset,label_dict,e,image_dataset):
    for i in range(len(model_object_list)):
        label_dict = iniitalize_dict(lable_dataset)

        image_dataset = auto_reshape_images(model_object_list[i].img_shape, image_dataset)

        right, wrong = iterate_trough_imgs(model_object_list[i], image_dataset, lable_dataset,label_dict)

        percent = (right / (wrong + right)) * 100

        print(f"Right: {right}, wrong: {wrong}, percent correct: {percent}")

        get_model_results(label_dict, model_object_list[i], (e, True))

def iterate_trough_imgs(model_object_list,image_dataset,lable_dataset, label_dict): #image[i]
    right = 0
    wrong = 0
    
    data_len = len(image_dataset)
    progress = trange(data_len, desc='Image stuff', leave=True)
    
    for j in progress:
        progress.set_description(f"Image {j + 1} / {data_len}")
        progress.refresh()

        prediction = make_prediction(model_object_list.model, image_dataset[j].copy(),  model_object_list.get_size_tuple(3))
        predicted_label = np.argmax(prediction)

        if int(predicted_label) == int(lable_dataset[j]):
            right += 1
            label_dict[lable_dataset[j]][1] += 1
            # img.save(f"{path}/right/{i}_{predicted_label}_{int(100*np.max(softmaxed))}_{e}.jpg")
        else:
            wrong += 1
            label_dict[lable_dataset[j]][0] += 1
            # img.save(f"{path}/wrong/{i}_{lable_dataset[j]}_{predicted_label}_{int(100*np.max(softmaxed))}_{e}.jpg")
    return right,wrong


def update_values(key, label_dict, prt):
    class_name = str(key)
    right_name = str(label_dict[key][1]).rjust(4, ' ')
    wrong_name = str(label_dict[key][0]).rjust(4, ' ')
    class_size = label_dict[key][0]+label_dict[key][1]
    class_percent = (label_dict[key][1]/class_size)*100
    
    if prt:
        print(f"class: {class_name.zfill(2)} | right: {right_name} | wrong: {wrong_name} | procent: {round(class_percent, 2)}")
    
    return class_name, class_percent, class_size

def get_model_results(lable_dict, model_object ,settings,should_print=True):
    for key in lable_dict.keys():
        class_name, class_percent, class_size = update_values(key, lable_dict,should_print)
        model_object.csv_data.append([settings[0], model_object.get_size(), class_name, class_percent, class_size])

def quick():
    lazy_split = 1

    test_path = get_h5_test()
    train_path = get_h5_train()

    run_experiment_one(lazy_split, train_path, test_path, epochs=(1,4))

quick()