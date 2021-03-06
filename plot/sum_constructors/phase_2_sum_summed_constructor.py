import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from sum_constructors.sum_constructor_object import sum_constructor

def get_object(get_sub_category, get_category, get_class_accuracy):
    def get_headers(headers):
        headers_list = [['category', headers[2]]]
        headers_list[-1].extend(headers[3:])
        return headers_list

    def get_current_data(row):
        current_class_accuracy = [row[3:]]
        current_total_in_class = [int(float(row[2]))]
        current_category = row[0]
        
        return current_class_accuracy, current_total_in_class, current_category

    def calculate_class_accuracy(data):
        class_accuracy = []
        for i in range(len(data[0][0])):
            class_accuracy.append(get_class_accuracy([x[i] for x in data[0]] , data[1]))
        return class_accuracy

    def get_data_to_append(data, class_accuracy):
        return_list = [data[2], sum(data[1])]
        return_list.extend(class_accuracy)
        return return_list

    def add_to_current_data(data, row):
        data[0].append(row[3:])
        data[1].append(int(float(row[2])))

    def category_changed(data, row):
        return data[2] != row[0]

    return sum_constructor(get_headers, 
                           get_current_data, 
                           calculate_class_accuracy,
                           get_data_to_append,
                           add_to_current_data,
                           category_changed
                           )