import Noise_Generators.noise_main as noise
from Noise.generators.noise_main import Filter,premade_single_filter,apply_multiple_filters
from Dataset.load_h5 import h5_object
from global_paths import get_test_model_paths, get_paths, get_h5_test, get_h5_train
from Models.test_model import partial_accumilate_distribution, print_accumilate_distribution, make_prediction
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def phase_2_1(model, h5path, lazysplit, image_size):
    h5 = h5_object(h5path, training_split=dataset_split)
    values = []
    for j in range(lazy_split):
        original_images, original_labels, _, _ = h5_obj.shuffle_and_lazyload(j, lazy_split) #TODO need variant of this that does not generate test set or shuffle
        image_tuples = add_noise((original_images,original_labels)) #tuple(image,class,filter)
        for i in range(len(image_tuples)):
            prediction = make_prediction(model, image_tuples[i][0], (image_size[0], image_size[1], 3))
            predicted_label = np.argmax(prediction) #Get the class with highest liklyhood of the predictions
            image_tuples[i] = image_tuples[i]+(predicted_label) #concatanate two tuples to create new tuple , which replacess the old one
        values.extend(image_tuples)
    convert_to_csv('phase_two/phase2_results.csv',headers,values) #tuple(image,class,filter,predicted_class)

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

def convert_to_csv(path,headers,values):
    with open(path, mode='w') as phase2_results:
        phase2_writer = csv.writer(phase2_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        phase2_writer.writerow(list(headers))
        
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
    newdatapoint = []
    with open('phase_two/phase2_results.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        groups = group_by_feature(header,csv_reader,'filter')
        for group in groups:
            classes = group_by_feature(header,group,'class')
            for _class in classes:
                error = calculate_error(_class)
                newdatapoint.append((_class,group,error)) #(class,filter,error)
            convert_to_csv(f'phase_two/phase2_{group}.csv', ['class','filters','error'], newdatapoint)

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
    for i in range(len(_classes))
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

    large_model_path, medium_model_path, small_model_path,belgium_model_path = get_test_model_paths()
        
    #large_model = tf.keras.models.load_model(large_model_path)
    #medium_model = tf.keras.models.load_model(medium_model_path)
    # small_model = tf.keras.models.load_model(small_model_path)
    belgium_model = tf.keras.models.load_model(belgium_model_path)
    phase_2_1(belgium_model,,10,(32,32))

