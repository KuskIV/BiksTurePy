import csv
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from global_paths import get_h5_train, get_h5_test, get_paths,  get_category_csv
from Dataset.load_h5 import h5_object
from plot.write_csv_file import cvs_object
from Dataset.dot import read_class_descrip, get_category_from_class

def get_sub_category(class_name):
    return get_category_from_class(subcategories, int(class_name) + 1)

def get_category(class_name):
    return get_category_from_class(categories, int(class_name) + 1)

class_text_labels, categories, subcategories = read_class_descrip(get_category_csv())
subcategories.append(categories[0]) #add danger as sub category
subcategories.append(categories[-1]) #add others as sub category

def iterate_through_dict(class_dict, output_dict, input_key):
    for key, value in class_dict.items():
        if not key in output_dict:
            output_dict[key] = {}
        output_dict[key][input_key] = value

def make_train_test_size_graph(save_path):
    test_path = get_h5_test()
    train_path = get_h5_train()
    
    train_Key = 'train'
    test_key = 'test'
    
    result_list = [['class', 'train', 'test']]
    
    h5_train = h5_object(train_path, training_split=1)
    h5_test = h5_object(test_path, training_split=1)
    
    csv_dict = {}
    
    iterate_through_dict(h5_train.images_in_classes, csv_dict, train_Key)    
    iterate_through_dict(h5_test.images_in_classes, csv_dict, test_key)    
    
    csv_list = list(csv_dict.items())
    
    csv_list.sort(key=lambda x: int(x[0]))
    
    for dick in csv_list:
        result_list.append([dick[0], dick[1][train_Key] if train_Key in dick[1] else 0, dick[1][test_key] if test_key in dick[1] else 0])
    
    csv_obj = cvs_object(save_path)
    csv_obj.write(result_list)

def sum_to_list(csv_data, current_category, current_train, current_test,current_sub_category=None):
    if current_sub_category != None:
        csv_data.append([current_category, current_sub_category, current_train, current_test])
    else:
        csv_data.append([current_category, current_train, current_test])
        
def sum_test_train_graph(input_path, base_path):
    save_path = f"{base_path}/sum_train_test_sub_cat.csv"
    f = open(input_path, 'r')
    reader = csv.reader(f)
    headers = next(reader, None)
    
    csv_data = [['category', 'sub category', headers[1], headers[2]]]
    
    with open(input_path, 'r') as csvfile:
        first_row = next(reader, None)
        current_sub_category = get_sub_category(first_row[0])
        current_category = get_category(first_row[0])
        
        current_train = int(first_row[1])
        current_test = int(first_row[2])
        
        plots = csv.reader(csvfile, delimiter=',')
        
        next(plots)
        next(plots)
        
        for row in plots:
            if current_sub_category != get_sub_category(row[0]):
                sum_to_list(csv_data, current_category, current_train, current_test, current_sub_category)
                
                if current_category != get_category(row[0]):
                    current_category = get_category(row[0])
                    
                current_sub_category = get_sub_category(row[0])
                current_train = int(row[1])
                current_test = int(row[2])
            else:
                current_train += int(row[1])
                current_test += int(row[2])
        sum_to_list(csv_data, current_category, current_train, current_test, current_sub_category)
        csv_obj = cvs_object(save_path)
        csv_obj.write(csv_data)
        return save_path
    
    
def sum_summed_test_train_graph(input_path, save_path):
    save_path = f"{base_path}/sum_summed_train_test_sub_cat.csv"
    f = open(input_path, 'r')
    reader = csv.reader(f)
    headers = next(reader, None)
    
    csv_data = [['category', headers[2], headers[3]]]
    
    with open(input_path, 'r') as csvfile:
        first_row = next(reader, None)
        current_category = first_row[0]
        
        current_train = int(first_row[2])
        current_test = int(first_row[3])
        
        plots = csv.reader(csvfile, delimiter=',')
        
        next(plots)
        next(plots)
        
        for row in plots:
            if current_category != row[0]:
                sum_to_list(csv_data, current_category, current_train, current_test)

                current_category = row[0]
                current_train = int(row[2])
                current_test = int(row[3])
            else:
                current_train += int(row[2])
                current_test += int(row[3])
        sum_to_list(csv_data, current_category, current_train, current_test)
        csv_obj = cvs_object(save_path)
        csv_obj.write(csv_data)

def generate_both_summed():
    base_path = get_paths('phase_one_csv')
    train_test_path = f"{base_path}/train_test_dist.csv"
    make_train_test_size_graph(train_test_path)
    sum_train_test_path = sum_test_train_graph(train_test_path, base_path)
    sum_summed_test_train_graph(sum_train_test_path, base_path)