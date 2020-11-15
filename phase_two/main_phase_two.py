import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from operator import itemgetter

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from Noise_Generators.noise_main import Filter,premade_single_filter,apply_multiple_filters
from Dataset.load_h5 import h5_object
from global_paths import get_paths, get_h5_test, get_h5_train, get_phase_two_csv
from Models.test_model import partial_accumilate_distribution, print_accumilate_distribution, make_prediction
from phase_one.find_ideal_model import get_satina_gains_model_object_list
from general_image_func import auto_reshape_images,changeImageSize,rgba_to_rgb,convert_between_pill_numpy
from plot.write_csv_file import cvs_object
from plot.sum_for_model import sum_phase_2_files
from error_handler import check_if_valid_path as civp

def append_pred_to_tuples(image_tuples:list, model:object)->list:
    values = [("image","filter","class","predicted_class")]#headers for the csv that will be generated
    for i in range(len(image_tuples)):
        prediction = make_prediction(model.model, image_tuples[i][0], (model.img_shape[0], model.img_shape[1], 3))
        predicted_label = np.argmax(prediction) #Get the class with highest liklyhood of the predictions
        image_tuples[i] = (image_tuples[i]+tuple([predicted_label,'yeet'])) #concatanate two tuples to create new tuple , which replacess the old one
    values.extend(image_tuples)
    return values

def weird_shuffle_of_the_tuple(image_tuples:list,numpy_imgs:list)->list:
    for i in range(len(numpy_imgs)):
        image_tuples[i] = list(image_tuples[i])
        image_tuples[i][0] = numpy_imgs[i]
        image_tuples[i] = tuple(image_tuples[i])
    return image_tuples

def phase_2_1(model:object,noise_filter:dict, base_path:str, original_images:list, original_labels:list)->None:
    image_tuples = add_noise((convert_between_pill_numpy(original_images * 255,mode='numpy->pil'),original_labels),noise_filter) #tuple(image,class,filter) the method returns the before show tuple where some of the images ahve been applied some noise
    numpy_imgs = convert_between_pill_numpy([changeImageSize(model.img_shape[0],model.img_shape[1],im[0].convert('RGB')) for im in image_tuples],mode='pil->numpy')#?confused about the specefics, but the result seems to be a list of numpy imgs that is returned
    image_tuples = weird_shuffle_of_the_tuple(image_tuples,numpy_imgs) #!still unsure id this is nessesary
    values = append_pred_to_tuples(image_tuples,model)
    convert_to_csv(f"{base_path}/{model.get_csv_name()}_result.csv",[val[1:4] for val in values]) #tuple(image,class,filter,predicted_class)

def load_filters()->dict: #*DONE
    F = premade_single_filter('fog')
    R = premade_single_filter('rain')
    S = premade_single_filter('snow')
    D = premade_single_filter('day')
    N = premade_single_filter('night')
    dict = [{'fog':F}, {'night':N},{'rain':R},{'snow':S},{'day':D},{'night':N}]
    #dict = {'fog':F,'rain':R,'snow':S,'day':D,'night':N}
    return dict

def add_noise(imgs:list,noise_filter:dict)->list: #*DONE
    return apply_multiple_filters(imgs,filters = noise_filter, mode='rand', KeepOriginal=True)

def convert_to_csv(path:str,values:list)->None: #*DONE
    with open(path, mode='w') as phase2_results:
        phase2_writer = csv.writer(phase2_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for value in values:
            phase2_writer.writerow(list(value))

def calculate_error(_class:list)->float:#*DONE
    wrong,right = 0,0
    for c in _class:
        if c[1] == c[2]:
            right += 1
        else:
            wrong += 1
    return wrong + right, round((right / (wrong + right)) * 100, 2)

def find_feature_colume(headers:list,feature_lable:str)->int: #*DONE
    for i in range(len(headers)):
        if headers[i] == feature_lable:
            return i

def group_by_feature(header:list,csv_reader:object,feature_lable:str)->dict:
    groups = {}
    colum_num = find_feature_colume(header,feature_lable)
    for row in csv_reader:
        if row[colum_num] in groups.keys():
            groups[row[colum_num]].append(row)
        else:
            groups[row[colum_num]] = [row]
    return groups

def merge_csv(filter_names, saved_path, class_size_dict, model_names, base_path):
    class_dict = {}

    for name in filter_names:
        for model_name in model_names:
            with open(f"{base_path}/{name}_{model_name}.csv", 'r') as read_obj:
                reader = csv.reader(read_obj)
                data = list(reader)
                data[0][2] = f"{name}{model_name}"
                for row in data:
                    if not row[0] in class_dict:
                        class_dict[row[0]] = [row[0]]
                    class_dict[row[0]].append(row[2])

                    if name == filter_names[-1] and model_name == model_names[-1] and row[0] in class_size_dict:
                        class_dict[row[0]].append(class_size_dict[row[0]])
                    elif name == filter_names[-1] and model_name == model_names[-1]:
                        class_dict[row[0]].append('images')

    list_data = [class_dict[key] for key in class_dict.keys()]
    sort_list = list_data[1:]
    list_data = [list_data[0]]
    sort_list.sort(key= lambda sort_list: int(sort_list[0]))
    list_data.extend(sort_list)
    csv_obj = cvs_object(saved_path)
    csv_obj.write(list_data)

def complete_newdatapoints(data_points:list, images_in_classes:list, group:list)->list: #*DONE
    new_points = []
    for key, value in images_in_classes.items():
        if key not in [x[0] for x in data_points]:
            new_points.append([key,group,0])
    return new_points

def newdatapoint_creation_from_classes(classes:list,group:list,newdatapoint:list)->list:
    for _class in classes:
        size, error = calculate_error(classes[_class])
        newdatapoint.append((_class, group, error))
    return newdatapoint

def Generate_newdatapoints(base_path:str, model_name:str ,header:list, groups:dict, images_in_classes:list)->list:
    newdatapoint = [('class','filters','error')]
    filter_names = []
    for group in groups: #TODO unsure about some of the changes made, if error orrcures, propeply here
        classes = group_by_feature(header,groups[group],'class')
        newdatapoint = newdatapoint_creation_from_classes(classes,group,newdatapoint)
        newdatapoint.extend(complete_newdatapoints(newdatapoint, images_in_classes, group))
        convert_to_csv(f"{base_path}/{group}_{model_name}.csv", newdatapoint)
        filter_names.append(group)
        newdatapoint = [('class','filters','error')]
    return filter_names

def create_csv_to_plot(model_name:str, base_path:str, images_in_classes:list)->list: #*DONE
    csv_path = f"{base_path}/{model_name}_result.csv"
    
    with open(civp(csv_path), 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        header = next(csv_reader)
        groups = group_by_feature(header,csv_reader,'filter')
        filter_names = Generate_newdatapoints(csv_path, model_name, header, groups, images_in_classes)
    return filter_names

def initailize_initial_values(folder_extension:str)->tuple:#*DONE
    filters = load_filters()
    filter_names = []
    base_path = get_paths('phase_two_csv') if folder_extension == None else f"{get_paths('phase_two_csv')}/{folder_extension}"
    if not folder_extension == None and not os.path.exists(base_path):
        os.mkdir(base_path)

    return filters, base_path

def get_h5_with_models(h5_path:str, training_split:int=1, get_models:list=get_satina_gains_model_object_list, model_paths:str=None)->tuple:#*DONE
    h5_obj = h5_object(civp(h5_path), training_split=training_split)
    model_object_list = get_models(h5_obj.class_in_h5, load_trained_models=True, model_paths=model_paths)
    return h5_obj,model_object_list

def evaluate_models_on_noise(filters:list, model_objs:list,h5_obj:object,base_path:str)->list:#*DONE
    filter_names = []
    for _filter in filters:
        original_images, original_labels, _, _ = h5_obj.shuffle_and_lazyload(0, 1)
        for model_object in model_objs:
            phase_2_1(model_object, _filter, base_path, original_images, original_labels)
            filter_names.extend(create_csv_to_plot(model_object.get_csv_name(), base_path, h5_obj.images_in_classes))
    return filter_names

def generate_csv_files_for_phase2(filter_names:list, h5_obj:object, base_path:str)->None: #*DONE
    merge_csv_path = f"{base_path}/merged_file.csv"
    merge_csv(list(dict.fromkeys(filter_names)), merge_csv_path, h5_obj.images_in_classes, [x.get_csv_name() for x in model_object_list], base_path)
    sum_phase_2_files(base_path)

def ex_two_eval_noise(test_path:str, folder_extension:str, get_models:list=get_satina_gains_model_object_list, training_split:int=1, model_paths:str=None)->None:#*DONE
    filters, base_path = initailize_initial_values(folder_extension)
    h5_obj, model_object_list = get_h5_with_models(civp(test_path),training_split=training_split,get_models=get_models, model_paths=model_paths)
    filter_names = evaluate_models_on_noise(filters, model_object_list, h5_obj, base_path)
    generate_csv_files_for_phase2(filter_names,h5_obj,base_path) #TODO move all csv related function into this method, speceficly the one from 2_1    
    
