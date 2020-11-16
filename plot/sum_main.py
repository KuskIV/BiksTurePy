
from sum_constructors.sum_constructor_object import sum_constructor

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from Dataset.dot import read_class_descrip, get_category_from_class
from global_paths import get_paths, get_category_csv
from plot.phase2_merged_file_sum import sum_merged_file
from plot.test_train_sizes import sum_train_test_file

class_text_labels, categories, subcategories = read_class_descrip(get_category_csv())
subcategories.append(categories[0])
subcategories.append(categories[-1])

def get_sub_category(class_name):
    return get_category_from_class(subcategories, int(class_name))

def get_category(class_name):
    return get_category_from_class(categories, int(class_name))

def get_class_accuracy(current_class_accuracy, current_total_in_class):
    total = 0
    right = 0

    for i in range(len(current_class_accuracy)):
        total += int(current_total_in_class[i])
        right += int(current_total_in_class[i]) * (float(current_class_accuracy[i]) / 100)

    return (right / total) * 100


if __name__ == '__main__':
    sum_train_test_file(get_category, get_sub_category, get_class_accuracy)