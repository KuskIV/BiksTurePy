import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tqdm import trange
import sys, os
import csv
import os.path
from os import path
from matplotlib import pyplot as plt

from find_ideal_model import train_and_eval_models_for_size, get_belgian_model_object_list, get_satina_gains_model_object_list

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
from plot.sum_for_model import sum_for_model, sum_for_class_accuracy, sum_summed_for_class_accuracy


def find_ideal_model(h5_obj:object, model_object_list:list, epochs:int=10, lazy_split:int=10, save_models:bool=False)->None:
    """Will based on a list of model objects, a h5py file an a max epochs amount, train the models and record the accracy in order to find the
    best model

    Args:
        h5_obj (object): The training and validation set
        model_object_list (list): The list of model objects to train
        epochs (int, optional): The upper limit of how many epocs each model should trian for. Defaults to 10.
        lazy_split (int, optional): How many splits the training should be split into. Defaults to 10.
        save_models (bool, optional): A boolean value representing whether or not the models should be saved. Defaults to False.
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
            if model_object_list[i].run_on_epoch(epochs):
                print(f"\n\nTraining model {i + 1} / {len(model_object_list) } for part in dataset {j + 1} / {lazy_split}")
                validation_loss, validation_accuracy = train_and_eval_models_for_size(model_object_list[i].img_shape, model_object_list[i].model, train_images, train_labels, test_images, test_labels, epochs)
                
                for j in range(len(validation_loss)):
                    model_object_list[i].fit_data.append([j+1, validation_loss[j], validation_accuracy[j]])
            else:
                print(f"\n\nMESSAGE: For epoch {epochs} model {model_object_list[i].get_csv_name()} will not train anymore, as the limit is {model_object_list[i].epoch}")
        
        del(train_images)
        del(test_images)
        
    if save_models:
        for models in model_object_list:
            store_model(models.model, models.path)

def get_best_models(model_object_list:list)->list:
    """Will iterate through the csv file produced, and find at what epoch each model had the higest accuracy.
    The model name will be returned, in addition to the amount of epocs and the accuracy

    Args:
        model_object_list (list): The list of model objects

    Returns:
        list: A list of tupes consisting of the accuracy, epoch and the resolution
    """
    best_models = []

    for model_object in model_object_list:
        if not path.exists(model_object.get_summed_csv_path()):
            print(f"\nThe following path does not exist: {model_object.get_summed_csv_path()}\nCode: plot.write_csv_file.py")
            sys.exit()

        with open(model_object.get_summed_csv_path(), 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')

            next(plots)

            highest_accuracy = 0
            best_epoch = 0
            resolution = 0

            for row in plots:
                if float(row[1]) > highest_accuracy:
                    highest_accuracy = float(row[1])
                    best_epoch = row[0]
                    resolution = row[2]

        best_models.append((highest_accuracy, best_epoch, resolution))

    print("\nThe best epoch for each model is as follows:")
    for bm in best_models:
        print(f"    - model{bm[2]}_{bm[1]}, accuracy: {round(bm[0], 2)}")
    print("\n")

    return best_models


def generate_csv_for_best_model(best_model_names:list)->None:
    """Will based on the model object list produce a csv illustrating the accuracy for each epoch.
    This data is saved on the object when they are training

    Args:
        best_model_names (list): The input list of model objects
    """

    model_names = [f"model{x[2]}_{x[1]}" for x in best_model_names]
    model_indexes = [0]
    data = [['class']]
    data[0].extend(model_names)

    csv_base_path = get_paths('phase_one_csv')
    csv_path = f"{csv_base_path}/class_accuracy.csv"
    save_path = f"{csv_base_path}/class_accuracy_minimized.csv"

    if not path.exists(csv_path):
        print(f"\nThe following path does not exist: {csv_path}\nCode: plot.write_csv_file.py")
        sys.exit()

    with open(csv_path, 'r') as csv_obj:
        rows = csv.reader(csv_obj, delimiter=',')
        rows = list(rows)

        model_indexes.extend([rows[0].index(x) for x in model_names if x in rows[0]])

        for i in range(1, len(rows)):
            data.append([rows[i][x] for x in model_indexes])

    csv_obj = cvs_object(save_path)
    csv_obj.write(data)

def get_largest_index(best_model_names:list)->int:
    best_acc = 0
    best_index = 0

    for i in range(len(best_model_names)):
        if best_model_names[i][0] > best_acc:
            best_acc = best_model_names[i][0]
            best_index = i

    return best_index

def max_epoch_from_list(epoch_list):
    best_epoch = 0
    
    for i in range(1, len(epoch_list)):
        best_epoch = int(epoch_list[i][1]) if int(epoch_list[i][1]) > best_epoch else best_epoch
    
    return best_epoch

def sum_summed_plots(model_object_list:list)->None:
    csv_data = []
    raw_data = []
    
    for model_object in model_object_list:
        if not path.exists(model_object.get_summed_csv_path()):
            print(f"ERROR: the file \"{model_object.get_summed_csv_path()}\" does not exists when trying to sum it. Program will exit.")
            sys.exit()
        with open(model_object.get_summed_csv_path(), 'r') as csv_obj:
            rows = csv.reader(csv_obj, delimiter=',')
            rows = list(rows)
            
            if not len(rows) > 0:
                print(f"ERROR: the file \"{model_object.get_summed_csv_path()}\" only has {len(rows)} items, should be {model_object.output_layer_size}")

            rows[0][1] = model_object.get_csv_name()
            
            raw_data.append(rows)
    
    csv_data = [x[0:1] for x in raw_data[0]]
    
    for i in range(len(raw_data)):
        for j in range(len(csv_data)):
            csv_data[j].append(raw_data[i][j][1])
        
    csv_obj = cvs_object(f"{get_paths('phase_one_csv')}/sum_summed.csv")
    csv_obj.write(csv_data)

def output_best_model_names(model_object_list):
    output_names = []
    # best_models.append((highest_accuracy, best_epoch, resolution))
    # ['epoch', 'validation_loss', 'validation_accuracy']
    
    for model_object in model_object_list:
        output_names.append([model_object.fit_data[-1][1], model_object.fit_data[-1][0], model_object.get_size()])
        
    return output_names

def run_experiment_one(lazy_split:int, train_h5_path:str, test_h5_path:str, get_models, epochs_end:int=10, dataset_split:int=0.7)->None:
    """This method runs experiment one, and is done in several steps:
            1. For each epoch to train for, the models are trained. After each epoch, the accuracy is saved on the object.
            2. When the training is done, all the data is saved in csv files as (Epochs,Resolution,Class,Class_Acuracy,Total_in_Class)
                Here the accuracy for all classes has its own row.
            3. Once this is done, rather than representing all classes in each epoch in seperate row, this is combined with one row for
                each epoch (Epochs,Model_accuracy,Resolution).
            4. All the summed files are now combined into one file, as (class, model[resolution 1]_[epoch 1], ... , model[resolution n]_[epoch n])
            5. Now for each model, the maximal accuracy is found, and the given epoch is saved
            6. Based on this information, three new models are made, with the idael epoch, representing the best possible models for this experiment.

    Args:
        lazy_split (int): How many pieces the dataset should be split into
        train_h5_path (str): The path for the trainig h5py
        test_h5_path (str): The path for the test h5py
        epochs_end (int, optional): The upper limit for how many epochs to train the modesl for. Defaults to 10.
        dataset_split (int, optional): The split between the training and validation set. Defaults to 0.7.
    """
    h5_train = h5_object(train_h5_path, training_split=dataset_split)
    h5_test = h5_object(test_h5_path, training_split=1)

    if h5_train.class_in_h5 != h5_test.class_in_h5:
        print(f"The input train and test set does not have matching classes {h5_train.class_in_h5} - {h5_test.class_in_h5}")
        sys.exit()

    model_object_list = get_models(h5_train.class_in_h5)

    find_ideal_model(h5_train, model_object_list, lazy_split=lazy_split, epochs=epochs_end, save_models=True)

    print(f"\n------------------------\nTraining done. Now evaluation will be made.\n\n")

    _, _, image_dataset, lable_dataset = h5_train.shuffle_and_lazyload(0, 1)

    iterate_trough_models(model_object_list, epochs_end, image_dataset, lable_dataset) #TODO This should not use h5_test, rather the h5_trainig and evaluation set.

    del image_dataset
    del lable_dataset

    for model in model_object_list:
        csv_obj = cvs_object(f"{get_paths('phase_one_csv')}/{model.get_csv_name()}_fitdata.csv")
        csv_obj.write(model.fit_data)

    data = [['epoch']]
    
    max_len = max([len(x.fit_data) for x in model_object_list])
    
    for model_object in model_object_list:
        for i in range(max_len):
            if i+1 > len(data):
                data.append([i])
            
            if i == 0:
                data[i].append(model_object.get_csv_name())
            else:
                if i >= len(model_object.fit_data):
                    data[i].append(' ')
                else:
                    data[i].append(model_object.fit_data[i][1])
    csv_obj = cvs_object(f"{get_paths('phase_one_csv')}/fitdata_combined.csv")
    csv_obj.write(data)
            
    
    # save_plot(model_object_list) #TODO: LINES BELOW THIS SHOULD NOT BE OUT-COMMENTED
    # sum_plot(model_object_list)
    # sum_summed_plots(model_object_list)
    # sum_class_accuracy(model_object_list)

    # best_model_names = get_best_models(model_object_list)
    # generate_csv_for_best_model(best_model_names)
    # model_object_list = get_models(h5_train.class_in_h5)
    # image_dataset, lable_dataset, _, _ = h5_test.shuffle_and_lazyload(0, 1)

    # for i in range(len(model_object_list)):
    #     model_object_list[i].set_epoch(best_model_names[i][1])

    # find_ideal_model(h5_train, model_object_list, lazy_split=lazy_split, epochs=max_epoch_from_list(best_model_names), save_models=True)

    # iterate_trough_models(model_object_list, -1, image_dataset, lable_dataset)
    
    save_plot(model_object_list)
    sum_plot(model_object_list)
    sum_summed_plots(model_object_list)
    path = sum_class_accuracy(model_object_list, h5_train.images_in_classes)
    data = sum_for_class_accuracy(cvs_object(path))
    csv_obj = cvs_object(f"{get_paths('phase_one_csv')}/sum_class_accuracy.csv")
    csv_obj.write(data)
    data = sum_summed_for_class_accuracy(csv_obj)
    csv_obj.write(data, path=f"{get_paths('phase_one_csv')}/sum_summed_class_accuracy.csv", overwrite_path=True)
    
    
    # pathacc = f"{get_paths('phase_one_csv')}/sum_class_accuracy.csv"
    # pathacc2 = f"{get_paths('phase_one_csv')}/sum_summed_class_accuracy.csv"

    # path = f"{get_paths('phase_one_csv')}/class_accuracy.csv"
    # data = sum_for_class_accuracy(cvs_object(path))
    # csv_obj = cvs_object(pathacc)
    # csv_obj.write(data)
    # data = sum_summed_for_class_accuracy(csv_obj)
    # csv_obj.write(data, path=pathacc2, overwrite_path=True)

    # best_model_names = get_best_models(model_object_list)
    # generate_csv_for_best_model(best_model_names)
    
    # best_model_names = output_best_model_names(model_object_list)
    
    # generate_csv_for_best_model(best_model_names)
    # model_object_list = get_models(h5_train.class_in_h5)
    image_dataset, lable_dataset, _, _ = h5_test.shuffle_and_lazyload(0, 1)

    # best_model_names = get_best_models_loss(model_object_list)
    # for i in range(len(model_object_list)):
        # model_object_list[i].set_epoch(best_model_names[i][1])

    # find_ideal_model(h5_train, model_object_list, lazy_split=lazy_split, epochs=max_epoch_from_list(best_model_names), save_models=True)

    iterate_trough_models(model_object_list, -1, image_dataset, lable_dataset)



def get_best_models_loss(model_object_list):
    best_models = []
    
    for model_object in model_object_list:
        resolution = model_object.get_size()
        
        min_list = [x[1] for x in model_object.fit_data[1:]]
        min_index = np.argmin(np.array(min_list))
        best_epoch = model_object.fit_data[min_index][0]
        best_loss = model_object.fit_data[min_index][1]
        
        best_models.append([best_loss, best_epoch, resolution])
        
    return best_models

def sum_class_accuracy(model_object_list:list, images_in_classes)->dict:
    """When training the accuracy for each class for each epoch is recorded. Here the sum of all accuracies for all classes for each epoch is summed together.

    Args:
        model_object_list (list): The list of

    Returns:
        dict: [description]
    """
    save_path = f"{get_paths('phase_one_csv')}/class_accuracy.csv"
    model_class_accuracy = {}

    for model_object in model_object_list:
        model_class_accuracy[model_object.get_csv_name()] = {}

        if not path.exists(model_object.get_csv_path()):
                print(f"\nThe following path does not exist: {model_object.get_csv_path()}\nCode: plot.write_csv_file.py")
                sys.exit()

        with open(model_object.get_csv_path(), 'r') as csvfile:
                plots = csv.reader(csvfile, delimiter=',')

                next(plots)
                for row in plots:
                    if not row[0] in model_class_accuracy[model_object.get_csv_name()]:
                        model_class_accuracy[model_object.get_csv_name()][row[0]] = {}
                    model_class_accuracy[model_object.get_csv_name()][row[0]][row[2]] = row[3]

    data_list = convert_dict_to_list(model_class_accuracy, images_in_classes)
    save_data_obj = cvs_object(save_path)
    save_data_obj.write(data_list)

    return save_path

def convert_dict_to_list(model_class_accuracy:dict, images_in_classes)->list:
    data_list = [['Class', 'Size']]
    for key, value in model_class_accuracy.items():
        for key2, value2 in model_class_accuracy[key].items():
            data_list[0].append(f"{key}_{key2}")
            keys = list(value2.keys())
            keys.sort(key=int)

            for i in range(len(keys)):
                if len(data_list) == i + 1:
                    data_list.append([keys[i], images_in_classes[keys[i]]]) #TODO HERE
                data_list[int(i) + 1].append(model_class_accuracy[key][key2][str(keys[i])])

    return data_list

def sum_plot(model_object_list:list)->None:
    csv_object_list =  []
    for model_object in model_object_list:
        obj = cvs_object(model_object.get_csv_path(), label=model_object.get_size())
        data = sum_for_model(obj)
        obj.write(data, model_object.get_summed_csv_path(), overwrite_path=True)
        csv_object_list.append(obj)
    # plot(csv_object_list)

def save_plot(model_object_list:list)->None:
    for model_object in model_object_list:
        cvs_obj = cvs_object(model_object.get_csv_path())
        cvs_obj.write(model_object.csv_data)

def iniitalize_dict(lable_dataset:list)->dict:
    label_dict = {}
    for lable in lable_dataset:
            label_dict[lable] = [0,0]
    return label_dict


def iterate_trough_models(model_object_list:list, e:int, image_dataset, lable_dataset)->None:
    """Iterates through the modesl to get the accuracy using the validation set. This information is saved on the object,
    and later saved in a csv file

    Args:
        model_object_list (list): The list of model objects
        label_dict (dict): A dictionary containing the data to plot in the csv file. One key per class
        e (int): The current epoch
        h5_test (object): The h5py object containg the images
    """
    update_epoch = True if e == -1 else False
    
    for i in range(len(model_object_list)):
        if update_epoch:
            e = model_object_list[i].fit_data[-1][0]
            
        if int(e) < 0:
            print(f"\nERROR: when iterating through the models, the epoch is smaller than 0 ({e})\n")
            sys.exit()
        
        label_dict = iniitalize_dict(lable_dataset)

        image_dataset = auto_reshape_images(model_object_list[i].img_shape, image_dataset)

        right, wrong = iterate_trough_imgs(model_object_list[i], image_dataset, lable_dataset,label_dict)

        percent = (right / (wrong + right)) * 100
        print(f"\nModel: \"{model_object_list[i].path.split('/')[-1].split('.')[0]}\"\nEpocs: {e} \nResult: \n    Right: {right}\n    wrong: {wrong}\n    percent correct: {percent}\n\n")

        get_model_results(label_dict, model_object_list[i], (e, True))

def iterate_trough_imgs(model_object:object,image_dataset:list,lable_dataset:list,label_dict:dict)->tuple:
    """For each image in the dataset, it is predicted using the input mode, and the result saved in the lable_dict

    Args:
        model_object (object): One model to use
        image_dataset (list): The image dataset to predict on the
        lable_dataset (list): The labes fitting the images
        label_dict (dict): The dict to save the data on

    Returns:
        tuple: The amount of rith and wrong guesses
    """

    right = 0
    wrong = 0

    data_len = len(image_dataset)
    progress = trange(data_len, desc='Image stuff', leave=True)

    for j in progress:
        progress.set_description(f"Image {j + 1} / {data_len} has been predicted")
        progress.refresh()

        prediction = make_prediction(model_object.model, image_dataset[j].copy(),  model_object.get_size_tuple(3))
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


def update_values(key:int,label_dict:dict,prt:bool)->tuple:
    """The accuracy and class size is calculated and returned as a tuple

    Args:
        key (int): The key for the current class in the dictionary
        label_dict (dict): The lable dict to get data from
        prt (bool): A boolean deciding whether or not to pring

    Returns:
        tuple: a typle containing the class name, accuracy and size, to be added to the class dict
    """
    class_name = str(key)
    right_name = str(label_dict[key][1]).rjust(4, ' ')
    wrong_name = str(label_dict[key][0]).rjust(4, ' ')
    class_size = label_dict[key][0]+label_dict[key][1]
    class_percent = round((label_dict[key][1]/class_size)*100, 2)

    if prt:
        print(f"class: {class_name.zfill(3)} | right: {right_name} | wrong: {wrong_name} | procent: {round(class_percent, 2)}")

    return class_name, class_percent, class_size

def get_model_results(lable_dict:dict, model_object:object ,settings:tuple,should_print:bool=True)->None:
    """Appends data to the model object, later to be inputtet into the csv file. This includ things regarding how accurate the model is and so on

    Args:
        lable_dict (dict): A dictionary containg information regarding the classes
        model_object (object): The current model to add informatino to
        settings (tuple): Settings include the current epoch
        should_print (bool, optional): A boolean representing wether or not to print. Defaults to True.
    """

    if should_print:
        print(f"----------------\n\nDetails regarding each class accuracy is as follows:\n")

    for key in lable_dict.keys():
        class_name, class_percent, class_size = update_values(key, lable_dict,should_print)
        model_object.csv_data.append([model_object.fit_data[-1][0], model_object.get_size(), class_name, class_percent, class_size])

    if should_print:
        print("----------------")

# def quick()->None:
#     lazy_split = 1

#     test_path = get_h5_test()
#     train_path = get_h5_train()

#     run_experiment_one(lazy_split, train_path, test_path, epochs_end=1)
