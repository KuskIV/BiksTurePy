import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from sum_constructors.sum_constructor_object import sum_constructor

def get_object(get_class_accuracy):
    def get_headers(headers):
        return [[headers[0], 'model accuracy', headers[1]]]

    def get_current_data(row):
        current_resolution = row[1]
        current_class_accuracy = [row[3]]
        current_total_in_class = [row[4]]
        current_epoc = row[0]

        return current_resolution, current_class_accuracy, current_total_in_class, current_epoc

    def calculate_class_accuracy(data):
        return get_class_accuracy(data[1], data[2])

    def get_data_to_append(data, class_accuracy):
        return [data[3], class_accuracy, data[0]]

    def add_to_current_data(data, row):
        data[1].append(row[3])
        data[2].append(row[4])

    def category_changed(data, row):
        return data[3] != row[0]

    return sum_constructor(get_headers, 
                           get_current_data, 
                           calculate_class_accuracy,
                           get_data_to_append,
                           add_to_current_data,
                           category_changed
                           )