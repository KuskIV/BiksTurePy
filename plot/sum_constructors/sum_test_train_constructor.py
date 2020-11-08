import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from sum_constructors.sum_constructor_object import sum_constructor

def get_object(get_sub_category, get_category, get_class_accuracy):
    def get_headers(headers):
        headers_list = [['category', 'sub category', headers[1], headers[2]]]
        return headers_list

    def get_current_data(row):
        current_sub_category = get_sub_category(row[0])
        current_category = get_category(row[0])
        current_train = int(row[1])
        current_test = int(row[2])
        return [current_sub_category, current_category, current_train, current_test]

    def calculate_class_accuracy(data):
        return 0

    def get_data_to_append(data, class_accuracy):
        return_list = [data[1], data[0], data[2], data[3]]
        return return_list

    def add_to_current_data(data, row):
        data[2] += int(row[1])
        data[3] += int(row[2])

    def category_changed(data, row):
        return data[0] != get_sub_category(row[0])

    return sum_constructor(get_headers, 
                           get_current_data, 
                           calculate_class_accuracy,
                           get_data_to_append,
                           add_to_current_data,
                           category_changed
                           )