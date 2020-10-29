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
from phase_one.find_ideal_model import get_belgian_model_object_list
from global_paths import  get_h5_test, get_h5_train
from general_image_func import auto_reshape_images,changeImageSize,rgba_to_rgb
from plot.write_csv_file import cvs_object


def phase_2_1(model, h5path, lazy_split, image_size,noise_filter, dataset_split=1):
    h5_obj = h5_object(h5path, training_split=dataset_split)
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
    convert_to_csv('phase_two/csv_output/phase2_results.csv',[val[1:4] for val in values]) #tuple(image,class,filter,predicted_class) #TODO @Jeppe, fix, dont hardcode this path

def normalize_and_convert(img:np.array):
    img = img # * 255.0
    img = img.astype(np.uint8)
    return Image.fromarray(img)

def convert_between_pill_numpy(imgs,mode):
    if mode == 'pil->numpy':
        return [np.asarray(im) for im in imgs]
    if mode == 'numpy->pil':
        return [normalize_and_convert(im) for im in imgs]


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

def calculate_error(_class):#TODO simnple calculation finding the succes.
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

def generate_csv_name(filter_name):
    return f'phase_two/csv_output/phase2_{filter_name}.csv'

def merge_csv(filter_names, saved_path):
    class_dict = {}

    for name in filter_names:
        with open(generate_csv_name(name), 'r') as read_obj:
            reader = csv.reader(read_obj)
            data = list(reader)
            data[0][2] = name

            for row in data:
                if not row[0] in class_dict:
                    class_dict[row[0]] = [row[0]]
                class_dict[row[0]].append(row[2])
                if len(class_dict[row[0]]) > 2 and row[0] != 'class':
                    class_dict[row[0]][-1] = round(float(class_dict[row[0]][-1]) - float(class_dict[row[0]][1]), 2)

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
    with open('phase_two/csv_output/phase2_results.csv', 'r') as read_obj:#TODO @Jeppe, fix, dont hardcode this path
        csv_reader = csv.reader(read_obj)
        header = next(csv_reader)
        groups = group_by_feature(header,csv_reader,'filter')
        for group in groups:
            classes = group_by_feature(header,groups[group],'class')
            for _class in classes:
                size, error = calculate_error(classes[_class])
                newdatapoint.append((_class, group, error)) #(class,filter,error)
            convert_to_csv(generate_csv_name(group), newdatapoint)
            filter_names.append(group)
            newdatapoint = [('class','filters','error')]
    return filter_names

def QuickDebug():
    models = get_belgian_model_object_list(63, load_trained_models=True) # TODO: fix, dont hardcode class count
    test_path = get_h5_test()
    filters = load_filters()
    filter_names = []
    for n_filter in filters:
        phase_2_1(models[2], test_path,1,models[2].img_shape, n_filter)
        filter_names.extend(create_csv_to_plot())
    merge_csv(list(dict.fromkeys(filter_names)), generate_csv_name('merged_file'))

QuickDebug()

