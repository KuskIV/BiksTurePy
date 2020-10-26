import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from Noise_Generators.noise_main import Filter,premade_single_filter,apply_multiple_filters
from Dataset.load_h5 import h5_object
from global_paths import get_test_model_paths, get_paths, get_h5_test, get_h5_train
from Models.test_model import partial_accumilate_distribution, print_accumilate_distribution, make_prediction
from phase_one.find_ideal_model import get_model_object_list
from global_paths import  get_h5_test


def phase_2_1(model, h5path, lazy_split, image_size, dataset_split=1):
    h5_obj = h5_object(h5path, training_split=dataset_split)
    values = [("image","class","filter","predicted_class")]
    for j in range(lazy_split):
        original_images, original_labels, _, _ = h5_obj.shuffle_and_lazyload(j, lazy_split) #TODO need variant of this that does not generate test set or shuffle
        image_tuples = add_noise((convert_between_pill_numpy(original_images,mode='numpy->pil'),original_labels)) #tuple(image,class,filter)
        numpy_imgs = convert_between_pill_numpy(image_tuples[0],mode='pil->numpy')
        for i in range(len(numpy_imgs)):
            image_tuples[i] = list(image_tuples[i])
            image_tuples[i][0] = numpy_imgs[i]
            image_tuples[i] = tuple(image_tuples[i])

        for i in range(len(image_tuples)):
            prediction = make_prediction(model, image_tuples[i][0], (image_size[0], image_size[1], 3))
            predicted_label = np.argmax(prediction) #Get the class with highest liklyhood of the predictions
            image_tuples[i] = image_tuples[i]+(predicted_label) #concatanate two tuples to create new tuple , which replacess the old one
        values.extend(image_tuples)
    convert_to_csv('phase_two/phase2_results.csv',values) #tuple(image,class,filter,predicted_class)

def convert_between_pill_numpy(imgs,mode):
    lst = []
    if mode == 'pil->numpy':
        return [np.array(im) for im in imgs]
    if mode == 'numpy->pil':
        return [Image.fromarray(im) for im in imgs]


def load_filters():
    F = premade_single_filter('fog')
    R = premade_single_filter('rain')
    S = premade_single_filter('snow')
    D = premade_single_filter('day')
    N = premade_single_filter('night')
    dict = {'fog':F,'rain':R,'snow':S,'day':D,'night':N}
    return dict

def add_noise(imgs):
    return apply_multiple_filters(imgs,filters = load_filters(), mode='rand', KeepOriginal=True)

def convert_to_csv(path,values):
    with open(path, mode='w') as phase2_results:
        phase2_writer = csv.writer(phase2_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        for value in values:
            phase2_writer.writerow(list(value))

def calculate_error(_class):#TODO simnple calculation finding the succes.
    pass

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

def create_csv_to_plot():
    newdatapoint = [('class','filters','error')]
    with open('phase_two/phase2_results.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        groups = group_by_feature(header,csv_reader,'filter')
        for group in groups:
            classes = group_by_feature(header,group,'class')
            for _class in classes:
                error = calculate_error(_class)
                newdatapoint.append((_class,group,error)) #(class,filter,error)
            convert_to_csv(f'phase_two/phase2_{group}.csv', newdatapoint)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plot_phase2_result(path):
    filters = []
    _classes = []
    errors = []
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        for row in csv_reader:
            filters.append(row[1])
            _classes.append(row[0])
            errors.append(row[2])
    x = np.arange(len(filters))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    
    rects = []
    for i in range(len(_classes)):
        rects.append(ax.bar(x - width/2, errors[i], width, label=_classes[i]))
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Errors')
    ax.set_title('scores for the diffrent classes with noise')
    ax.set_xticks(x)
    ax.set_xticklabels(filters)
    ax.legend()

    for rect in rects:
        autolabel(rect)

    fig.tight_layout()

    plt.show()

def plot_phase2_results():
    csv_paths = []
    for path in csv_paths:
        plot_phase2_result(path)
    #matplotlib bar(), https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    
def QuickDebug():
    models = get_model_object_list(63)
    test_path = get_h5_test()

    phase_2_1(models[2], test_path,1,models[2].img_shape)

QuickDebug()

