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
from global_paths import get_test_model_paths, get_paths, get_h5_test, get_h5_train
from Models.test_model import partial_accumilate_distribution, print_accumilate_distribution, make_prediction
from phase_one.find_ideal_model import get_belgian_model_object_list, get_satina_gains_model_object_list
from global_paths import  get_h5_test, get_h5_train, get_phase_two_csv
from general_image_func import auto_reshape_images,changeImageSize,rgba_to_rgb,convert_between_pill_numpy
from plot.write_csv_file import cvs_object


def phase_2_1(model, h5_obj, lazy_split, image_size,noise_filter):
    values = [("image","filter","class","predicted_class")]
    lazy_load = 2
    for j in range(lazy_load):
        original_images, original_labels, _, _ = h5_obj.shuffle_and_lazyload(j, lazy_split) #TODO need variant of this that does not generate test set or shuffle

        image_tuples = add_noise((convert_between_pill_numpy(original_images * 255,mode='numpy->pil'),original_labels),noise_filter) #tuple(image,class,filter)
        numpy_imgs = convert_between_pill_numpy([changeImageSize(image_size[0],image_size[1],im[0].convert('RGB')) for im in image_tuples],mode='pil->numpy')
        
        #print(len(numpy_imgs))
        for i in range(len(numpy_imgs)):
            image_tuples[i] = list(image_tuples[i])
            image_tuples[i][0] = numpy_imgs[i]
            image_tuples[i] = tuple(image_tuples[i])

        for i in range(len(image_tuples)):
            prediction = make_prediction(model.model, image_tuples[i][0], (image_size[0], image_size[1], 3))
            predicted_label = np.argmax(prediction) #Get the class with highest liklyhood of the predictions
            image_tuples[i] = (image_tuples[i]+tuple([predicted_label,'yeet'])) #concatanate two tuples to create new tuple , which replacess the old one
        values.extend(image_tuples)
    convert_to_csv(get_phase_two_csv('results'),[val[1:4] for val in values]) #tuple(image,class,filter,predicted_class)

def load_filters():
    F = premade_single_filter('fog')
    R = premade_single_filter('rain')
    S = premade_single_filter('snow')
    D = premade_single_filter('day')
    N = premade_single_filter('night')
    dict = [{'fog':F}, {'night':N},{'rain':R},{'snow':S},{'day':D},{'night':N}]
    #dict = {'fog':F,'rain':R,'snow':S,'day':D,'night':N}
    return dict

def add_noise(imgs,noise_filter):
    return apply_multiple_filters(imgs,filters = noise_filter, mode='rand', KeepOriginal=True)

def convert_to_csv(path,values):
    with open(path, mode='w') as phase2_results:
        phase2_writer = csv.writer(phase2_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        for value in values:
            phase2_writer.writerow(list(value))

def calculate_error(_class):
    wrong = 0
    rigth = 0
    for c in _class:
        if c[1] == c[2]:
            rigth += 1
        else:
            wrong += 1
    return wrong + rigth, round((rigth / (wrong + rigth)) * 100, 2)

def find_feature_colume(headers:list,feature_lable:str)->int:
    for i in range(len(headers)):
        if headers[i] == feature_lable:
            return i
        
def group_by_feature(header,csv_reader,feature_lable:str):
    groups = {}
    colum_num = find_feature_colume(header,feature_lable)
    for row in csv_reader:
        if row[colum_num] in groups.keys():
            groups[row[colum_num]].append(row)
        else:
            groups[row[colum_num]] = [row]
    return groups

def merge_csv(filter_names, saved_path, class_size_dict):
    class_dict = {}

    for name in filter_names:
        with open(get_phase_two_csv(name), 'r') as read_obj:
            reader = csv.reader(read_obj)
            data = list(reader)
            data[0][2] = name

            for row in data:
                if not row[0] in class_dict:
                    class_dict[row[0]] = [row[0]]
                class_dict[row[0]].append(row[2])
                
                if name == filter_names[-1] and row[0] in class_size_dict:
                    class_dict[row[0]].append(class_size_dict[row[0]])
                elif name == filter_names[-1]:
                    class_dict[row[0]].append('images')
                    
                # if len(class_dict[row[0]]) > 2 and row[0] != 'class':
                    # class_dict[row[0]][-1] = round(float(class_dict[row[0]][-1]) - float(class_dict[row[0]][1]), 2)

    list_data = [class_dict[key] for key in class_dict.keys()]
    sort_list = list_data[1:]
    list_data = [list_data[0]]
    sort_list.sort(key= lambda sort_list: int(sort_list[0]))
    list_data.extend(sort_list)
    csv_obj = cvs_object(saved_path)
    csv_obj.write(list_data)

    # with open(saved_path, 'w') as write_obj:

def create_csv_to_plot():
    newdatapoint = [('class','filters','error')]
    filter_names = []
    with open(get_phase_two_csv('results'), 'r') as read_obj:#TODO @Jeppe, fix, dont hardcode this path | respond from jeppe 'no'
        csv_reader = csv.reader(read_obj)
        header = next(csv_reader)
        groups = group_by_feature(header,csv_reader,'filter')
        for group in groups:
            classes = group_by_feature(header,groups[group],'class')
            for _class in classes:
                size, error = calculate_error(classes[_class])
                newdatapoint.append((_class, group, error)) #(class,filter,error)
            convert_to_csv(get_phase_two_csv(group), newdatapoint)
            filter_names.append(group)
            newdatapoint = [('class','filters','error')]
    return filter_names

def sum_merged_csv(input_path:str, output_path:str)->None:
    ouput_data = [['category', 'subcategory', 'images']]
    
    if not os.path.exists(input_path):
        print(f"ERROR: the csv file \"{input_path}\" does not exists. The program will now execute")
        sys.exit()
        
    with open(input_path, 'r') as read_obj:
        reader = csv.reader(read_obj)
        data = list(reader)
        
        
        

def quick_debug():
    test_path = get_h5_test()
    filters = load_filters()
    filter_names = []
    trainin_split = 1
    
    h5_obj = h5_object(test_path, training_split=trainin_split)
    models = get_satina_gains_model_object_list(h5_obj.class_in_h5, load_trained_models=True)
    
    for n_filter in filters:
        phase_2_1(models[2],h5_obj,1,models[2].img_shape, n_filter)
        filter_names.extend(create_csv_to_plot())
    
    merge_csv(list(dict.fromkeys(filter_names)), get_phase_two_csv('merged_file'), h5_obj.images_in_classes)
    sum_merged_csv(get_phase_two_csv('merged_file'), get_phase_two_csv('summed_merged_file'))

quick_debug()

