import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from sum_constructors.sum_constructor_object import sum_constructor

def get_object(get_sub_category, get_category, get_class_accuracy):
    def get_headers(headers):
        return [['category', 'sub-category', headers[-1]]]

    def get_current_data(row):
        current_class_accuracy = [row[1:-1]]
        current_total_in_class = [int(row[-1])]
        current_category = get_category(row[0])
        current_sub_category = get_sub_category(row[0])
        
        return current_class_accuracy, current_total_in_class, current_category, current_sub_category

    def calculate_class_accuracy(data):
        class_accuracy = []
        for i in range(len(data[0][0])):
            class_accuracy.append(get_class_accuracy([x[i] for x in data[0]] , data[1]))
        return class_accuracy

    def get_data_to_append(data, class_accuracy):
        return_list = [data[2], data[3], sum(data[1])]
        return_list.extend(class_accuracy)
        return return_list

    def add_to_current_data(data, row):
        data[0].append(row[1:-1])
        data[1].append(int(row[-1]))

    def category_changed(data, row):
        return data[3] != get_sub_category(row[0])

    return sum_constructor(get_headers, 
                           get_current_data, 
                           calculate_class_accuracy,
                           get_data_to_append,
                           add_to_current_data,
                           category_changed
                           )