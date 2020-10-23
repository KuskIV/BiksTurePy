import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tqdm import trange

from main_phase_one import find_ideal_model
from find_ideal_model import get_processed_models, get_models

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from Dataset.load_h5 import h5_object
from global_paths import get_test_model_paths, get_paths, get_h5_test, get_h5_train
from plot.write_csv_file import cvs_object, plot
from general_image_func import auto_reshape_images, convert_numpy_image_to_image
from Models.test_model import make_prediction

def do_experiment_epochs(lazy_split, get_model_objects, train_h5_path, test_h5_path, epochs=(3,4), dataset_split=0.7):
    label_dict = {}

    big_boi_list = [['Epochs', 'Resolution', 'Class', 'Class_Acuracy', 'Total_in_Class']]

    h5_train = h5_object(train_h5_path, training_split=dataset_split)
    h5_test = h5_object(test_h5_path, training_split=1)

    model_object_list = get_model_objects()

    image_dataset, lable_dataset, _, _ = h5_test.shuffle_and_lazyload(0, 1)
    
    for e in range(epochs[0], epochs[1]):
        print(f"\n----------------\nRun {e} / {15}\n----------------\n")
        
        find_ideal_model(h5_train, model_object_list, lazy_split=lazy_split, epochs=e)
        
        print(f"\n------------------------\nTraining done. Now evaluation will be made, using {e} epochs.\n\n")

        big_boi_list.extend(iterate_trough_models(model_object_list, lable_dataset, label_dict, e, image_dataset))
        save_plot(big_boi_list)

def save_plot(big_boi_list):
    cvs_obj = cvs_object(f"{get_paths('phase_one_csv')}/big_boi.csv")
    cvs_obj.write(big_boi_list)
    plot([cvs_obj])

def iniitalize_dict(lable_dataset):
    label_dict = {}
    for lable in lable_dataset:
            label_dict[lable] = [0,0]
    return label_dict


def iterate_trough_models(model_object_list,lable_dataset,label_dict,e,image_dataset):
    returnlist = []
    for i in range(len(model_object_list)):
        label_dict = iniitalize_dict(lable_dataset)

        image_dataset = auto_reshape_images(model_object_list[i].img_shape, image_dataset)

        right, wrong = iterate_trough_imgs(model_object_list[i].model, image_dataset, lable_dataset,label_dict)

        percent = (right / (wrong + right)) * 100

        print(f"Right: {right}, wrong: {wrong}, percent correct: {percent}")

        returnlist.extend(get_model_results(label_dict,(e, model_object_list[i].get_shape()), True))
    return returnlist

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
    class_name = str(key).zfill(2)
    right_name = str(label_dict[key][1]).rjust(4, ' ')
    wrong_name = str(label_dict[key][0]).rjust(4, ' ')
    class_size = label_dict[key][0]+label_dict[key][1]
    class_percent = (label_dict[key][1]/class_size)*100
    
    if prt:
        print(f"class: {class_name} | right: {right_name} | wrong: {wrong_name} | procent: {round(class_percent, 2)}")
    
    return class_name, class_percent, class_size

def get_model_results(lable_dict,settings,should_print=True):
    result_list =[]
    for key in lable_dict.keys():
        class_name, class_percent, class_size = update_values(key, lable_dict,should_print)
    
        result_list.append([settings[0], settings[1], class_name, class_percent, class_size])
    return result_list

def quick():
    lazy_split = 1

    test_path = get_h5_test()
    train_path = get_h5_train()

    do_experiment_epochs(lazy_split, get_models, train_path, test_path)

quick()