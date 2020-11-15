import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tqdm import trange
import sys, os
import csv
import os.path
from os import path
from matplotlib import pyplot as plt

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from phase_one.find_ideal_model import train_and_eval_models_for_size, get_satina_gains_model_object_list
from phase_one.test_csv import combine_two_summed_class_accracy
from Dataset.load_h5 import h5_object
from Models.create_model import store_model
from global_paths import get_paths
from plot.write_csv_file import cvs_object, plot
from general_image_func import auto_reshape_images, convert_numpy_image_to_image
from Models.test_model import make_prediction
from plot.sum_for_model import sum_for_model, sum_for_class_accuracy, sum_summed_for_class_accuracy
from error_handler import check_if_valid_path, custom_error_check


def find_ideal_model(h5_obj:object, model_object_list:list, epochs:int=10, lazy_split:int=10, save_models:bool=False, data_to_test_on=1)->None:
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

    print(f"-----------------")
    print(f"Experiment one will now execute. This will be done using the following models:")
    for model_object in model_object_list:
        print(f"    - {model_object.get_csv_name()} ({model_object.path})")
    print(f"-----------------")

    train_images, train_labels, test_images, test_labels = h5_obj.shuffle_and_lazyload(0, data_to_test_on)

    print(f"Images in train_set: {len(train_images)} ({len(train_images) == len(train_labels)}), Images in val_set: {len(test_images)} ({len(test_images) == len(test_labels)})")

    for i in range(len(model_object_list)):
        if model_object_list[i].run_on_epoch(epochs):
            print(f"\n\nTraining model {i + 1} / {len(model_object_list)} for part in dataset {1} / {lazy_split}")
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
            print(f"A mode using resolution {models.img_shape} has successfully been saved at path \"{models.path}\"")

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
        check_if_valid_path(model_object.get_summed_csv_path())

        with open(model_object.get_summed_csv_path(), 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')

            next(plots)

            highest_accuracy = 0
            best_epoch = 0
            resolution = 0

            for row in plots:
                try:
                    if float(row[1]) > highest_accuracy:
                        highest_accuracy = float(row[1])
                        best_epoch = row[0]
                        resolution = row[2]
                except ValueError:
                    custom_error_check(False, f'{row[1]} cannot be converted to a float')
                except IndexError:
                    custom_error_check(False, f'Index two is being accesed in an array of lengt {len(row)}')

        best_models.append((highest_accuracy, best_epoch, resolution))

    print("\nThe best epoch for each model is as follows:")
    for bm in best_models:
        try:
            print(f"    - model{bm[2]}_{bm[1]}, accuracy: {round(bm[0], 2)}")
        except TypeError:
            custom_error_check(False, f"{bm[0]} cannot be rounded, as it is the wrong type")
        except IndexError:
            custom_error_check(False, f"index two is being accessed in an array of lenght {len(bm)}")
    print("\n")

    return best_models

def get_largest_index(best_model_names:list)->int:
    best_acc = 0
    best_index = 0

    for i in range(len(best_model_names)):
        try:
            if best_model_names[i][0] > best_acc:
                best_acc = best_model_names[i][0]
                best_index = i
        except TypeError:
            custom_error_check(False, f"{best_model_names[i][0]} cannot be comapted to a number using '>'.")
        except IndexError:
            custom_error_check(False, f"index zero is being accessed in an array of lenght {len(best_model_names[i])}")

    return best_index

def max_epoch_from_list(epoch_list):
    best_epoch = 0
    
    for i in range(1, len(epoch_list)):
        try:
            best_epoch = int(epoch_list[i][1]) if int(epoch_list[i][1]) > best_epoch else best_epoch
        except TypeError:
            custom_error_check(False, f"{epoch_list[i][1]} cannot be casted into a integer and compared to an int using '>'")
        except IndexError:
            custom_error_check(False, f"index one is being accessed in an array of lenght {len(epoch_list[i])}")
    
    return best_epoch

def verify_list_lenght(rows):
    return not len(rows) > 0

def sum_summed_plots(model_object_list:list, extension, base_path)->None:
    csv_data = []
    raw_data = []
    
    for model_object in model_object_list:
        check_if_valid_path(model_object.get_summed_csv_path(extension=extension))

        with open(model_object.get_summed_csv_path(extension=extension), 'r') as csv_obj:
            rows = csv.reader(csv_obj, delimiter=',')
            rows = list(rows)
            
            custom_error_check(not verify_list_lenght(rows), f"the file \"{model_object.get_summed_csv_path(extension=extension)}\" only has {len(rows)} items, should be {model_object.output_layer_size}")

            try:
                rows[0][1] = model_object.get_csv_name(extension=extension)
            except IndexError:
                custom_error_check(False, f"Cound not access index [0][1] of a list of lenght {len(rows)}")
            
            raw_data.append(rows)
    
    try:
        csv_data = [x[0:1] for x in raw_data[0]]
    except IndexError:
        custom_error_check(False, f"Cound not access index [0][0] of a list of lenght {len(raw_data)}")
    
    for i in range(len(raw_data)):
        for j in range(len(csv_data)):
            try:
                csv_data[j].append(raw_data[i][j][1])
            except IndexError:
                custom_error_check(False, f"Cound not access index raw_data[i][j][1]")
        
    csv_obj = cvs_object(f"{base_path}/{extension}_sum_summed.csv")
    csv_obj.write(csv_data)

def output_best_model_names(model_object_list):
    output_names = []
    
    for model_object in model_object_list:
        try:
            output_names.append([model_object.fit_data[-1][1], model_object.fit_data[-1][0], model_object.get_size()])
        except IndexError:
            custom_error_check(False, f"Cound not access index model_object.fit_data[-1][1]")
        
    return output_names

def iterate_and_sum(model_object_list, extension, sum_path, image_dataset, lable_dataset, epochs_end, images_in_classes, base_path, epochs=None):
    iterate_trough_models(model_object_list, epochs_end, image_dataset, lable_dataset, epochs=epochs)

    save_plot(model_object_list, extension)
    sum_plot(model_object_list, extension)
    sum_summed_plots(model_object_list, extension, base_path)
    path = sum_class_accuracy(model_object_list, images_in_classes, extension, base_path)
    data_class_acc_val = sum_for_class_accuracy(cvs_object(path))
    csv_obj = cvs_object(sum_path)
    csv_obj.write(data_class_acc_val)
    data = sum_summed_for_class_accuracy(csv_obj)
    csv_obj.write(data, path=f"{base_path}/{extension}_sum_summed_class_accuracy.csv", overwrite_path=True)

def verify_class_amounts(class_in_test, class_int_train):
    return class_in_test == class_int_train

def run_experiment_one(lazy_split:int, train_h5_path:str, test_h5_path:str, get_models, epochs_end:int=10, dataset_split:int=0.7, folder_extension = None, model_paths=None, data_to_test_on=1)->None:
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
    base_path = get_paths('phase_one_csv') if folder_extension == None else f"{get_paths('phase_one_csv')}/{folder_extension}"
    if not folder_extension == None and not os.path.exists(base_path):
        os.mkdir(base_path)
        
    h5_train = h5_object(train_h5_path, training_split=dataset_split)
    h5_test = h5_object(test_h5_path, training_split=1)

    custom_error_check(verify_class_amounts(h5_test.class_in_h5, h5_train.class_in_h5), f"The input train and test set does not have matching classes {h5_train.class_in_h5} - {h5_test.class_in_h5}")

    model_object_list = get_models(h5_train.class_in_h5, model_paths=model_paths)

    find_ideal_model(h5_train, model_object_list, lazy_split=lazy_split, epochs=epochs_end, save_models=True, data_to_test_on=data_to_test_on)

    print(f"\n------------------------\nTraining done. Now evaluation will be made.\n\n")

    sum_test_path = f"{base_path}/test_sum_class_accuracy.csv"
    sum_val_path = f"{base_path}/val_sum_class_accuracy.csv"

    model_object_list_loaded = get_models(h5_train.class_in_h5, load_trained_models=True)
    
    #TODO: Fix epoch count in test_val_sum_class_accuracy.csv
    _, _, image_dataset, lable_dataset = h5_train.shuffle_and_lazyload(0, data_to_test_on)
    iterate_and_sum(model_object_list, 'val', sum_val_path, image_dataset, lable_dataset, epochs_end, h5_train.images_in_classes, base_path)
    
    image_dataset, lable_dataset, _, _ = h5_test.shuffle_and_lazyload(0, data_to_test_on)
    iterate_and_sum(model_object_list_loaded, 'test', sum_test_path, image_dataset, lable_dataset, -1, h5_test.images_in_classes, base_path, epochs=[x.fit_data[-1][0] for x in model_object_list])
    combine_two_summed_class_accracy(sum_test_path, sum_val_path, base_path)

    save_fitdata(model_object_list, base_path)

def save_fitdata(model_object_list:list, base_path:str)->None:
    """This is the data used to produce the loss/epoch graht, and is saved in a csv file. as "epoch", "loss", "accuracy"

    Args:
        model_object_list (list): the list of model objects to save the data from
        base_path (str): the base path to save the data in
    """
    for model in model_object_list:
        csv_obj = cvs_object(f"{base_path}/{model.get_csv_name()}_fitdata.csv")
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
    csv_obj = cvs_object(f"{base_path}/fitdata_combined.csv")
    csv_obj.write(data)

def sum_class_accuracy(model_object_list:list, images_in_classes, extension, base_path)->dict:
    """When training the accuracy for each class for each epoch is recorded. Here the sum of all accuracies for all classes for each epoch is summed together.

    Args:
        model_object_list (list): The list of

    Returns:
        dict: [description]
    """
    save_path = f"{base_path}/{extension}_class_accuracy.csv"
    model_class_accuracy = {}

    for model_object in model_object_list:
        model_class_accuracy[model_object.get_csv_name()] = {}

        check_if_valid_path(model_object.get_csv_path(extension=extension))

        with open(model_object.get_csv_path(extension=extension), 'r') as csvfile:
                plots = csv.reader(csvfile, delimiter=',')

                next(plots)
                for row in plots:
                    try:
                        if not row[0] in model_class_accuracy[model_object.get_csv_name()]:
                            model_class_accuracy[model_object.get_csv_name()][row[0]] = {}
                        model_class_accuracy[model_object.get_csv_name()][row[0]][row[2]] = row[3]
                    except IndexError:
                        custom_error_check(False, f"Cannot access index three of row with a length of {len(row)}")

    data_list = convert_dict_to_list(model_class_accuracy, images_in_classes)
    save_data_obj = cvs_object(save_path)
    save_data_obj.write(data_list)

    return save_path

def convert_dict_to_list(model_class_accuracy:dict, images_in_classes:dict)->list:
    """Iterates through a dictionary of model class accuracies, and sorts them an converts them to a list which is returned

    Args:
        model_class_accuracy (dict): A dictionary of each class and its accruacy
        images_in_classes (dict): a dictionary of all classes and how many images it consists of

    Returns:
        list: the dictionary converted to a list
    """
    data_list = [['Class', 'Size']]
    for key, value in model_class_accuracy.items():
        for key2, value2 in model_class_accuracy[key].items():
            data_list[0].append(f"{key}_{key2}")
            keys = list(value2.keys())
            keys.sort(key=int)

            for i in range(len(keys)):
                if len(data_list) == i + 1:
                    data_list.append([keys[i], images_in_classes[keys[i]]])
                data_list[int(i) + 1].append(model_class_accuracy[key][key2][str(keys[i])])

    return data_list

def sum_plot(model_object_list:list, extension:str)->None:
    """converts the csv file showing the accuracy for each class, to a csv showing the accuracy for each sub category

    Args:
        model_object_list (list): a list of model objects which is iterated through
        extension (str): the extenson to add the the csv, in this case either being 'val' or 'test'
    """
    csv_object_list =  []
    for model_object in model_object_list:
        obj = cvs_object(model_object.get_csv_path(extension=extension), label=model_object.get_size())
        data = sum_for_model(obj)
        obj.write(data, model_object.get_summed_csv_path(extension=extension), overwrite_path=True)
        csv_object_list.append(obj)

def save_plot(model_object_list:list, extension)->None:
    """Iterates through each model object, and saves the accuracy for each class in a

    Args:
        model_object_list (list): the list of models to iterate through
        extension (str): the extension, deciding whether it is 'test' or 'val'
    """
    for model_object in model_object_list:
        cvs_obj = cvs_object(model_object.get_csv_path(extension=extension))
        cvs_obj.write(model_object.csv_data)

def initalize_dict(lable_dataset:list)->dict:
    """Creates a dictionary with a key for each different label, and the value being a tuple of (0, 0) wich represents right and wong guesses

    Args:
        lable_dataset (list): a list of all lables

    Returns:
        dict: a dictionary where each key is a label, and the values are tupes
    """
    label_dict = {}
    for lable in lable_dataset:
            label_dict[lable] = [0,0]
    return label_dict


def iterate_trough_models(model_object_list:list, e:int, image_dataset, lable_dataset, epochs=None)->None:
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
        if update_epoch and epochs != None:
            e = epochs[i]
            
        if int(e) < 0:
            print(f"\nERROR: when iterating through the models, the epoch is smaller than 0 ({e})\n")
            sys.exit()
        
        label_dict = initalize_dict(lable_dataset)

        image_dataset = auto_reshape_images(model_object_list[i].img_shape, image_dataset)

        right, wrong = iterate_trough_imgs(model_object_list[i], image_dataset, lable_dataset,label_dict)

        percent = (right / (wrong + right)) * 100
        
        print(f"\nModel: \"{model_object_list[i].path.split('/')[-1].split('.')[0]}\"\nEpocs: {e} \nResult: \n    Right: {right}\n    wrong: {wrong}\n    percent correct: {percent}\n\n")

        get_model_results(label_dict, model_object_list[i], (e, True), should_print=False)

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
        else:
            wrong += 1
            label_dict[lable_dataset[j]][0] += 1
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
