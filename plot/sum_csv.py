# from write_csv_file import cvs_object, plot
import os.path
from os import path
import sys, os
import csv

from write_csv_file import cvs_object

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from Dataset.dot import read_class_descrip, get_category_from_class

from global_paths import get_paths, get_category_csv

def get_class_accuracy(current_class_accuracy, current_total_in_class):
    total = 0
    right = 0

    for i in range(len(current_class_accuracy)):
        total += int(current_total_in_class[i])
        right += int(current_total_in_class[i]) * (float(current_class_accuracy[i]) / 100)

    return (right / total) * 100

def sum_csv(csv_obj):

    f = open(csv_obj.path, 'r')
    reader = csv.reader(f)
    headers = next(reader, None)

    new_cvs_file = [[headers[0], 'Model_accuracy', headers[1]]]

    with open(csv_obj.path, 'r') as csvfile:
            if not path.exists(csv_obj.path):
                print(f"\nThe following path does not exist: {csv_obj.path}\nCode: plot.write_csv_file.py")
                sys.exit()
            
            first_row = next(reader, None)
            
            current_resolution = first_row[1]
            current_class_accuracy = [first_row[3]]
            current_total_in_class = [first_row[4]]
            current_epoc = first_row[0]

            plots = csv.reader(csvfile, delimiter=',')

            next(plots)
            next(plots)
            for row in plots:
                if current_epoc != row[0]:
                    class_accuracy = get_class_accuracy(current_class_accuracy, current_total_in_class)
                    new_cvs_file.append([current_epoc, class_accuracy, current_resolution])

                    current_epoc = row[0]
                    current_class_accuracy = [row[3]]
                    current_total_in_class = [row[4]]
                    current_resolution = row[1]
                else:
                    current_class_accuracy.append(row[3])
                    current_total_in_class.append(row[4])
            class_accuracy = get_class_accuracy(current_class_accuracy, current_total_in_class)
            new_cvs_file.append([current_epoc, class_accuracy, current_resolution])
    
    return new_cvs_file

class_text_labels, categories, subcategories = read_class_descrip(get_category_csv())
subcategories.append(categories[0]) #add danger as sub category
subcategories.append(categories[-1]) #add others as sub category

def get_sub_category(class_name):
    return get_category_from_class(subcategories, int(class_name) + 1)

def get_category(class_name):
    return get_category_from_class(categories, int(class_name) + 1)


def sum_to_list(sub_cvs_file, current_category, current_class_accuracy, current_total_in_class, current_sub_category):
    class_accuracy = []
    for i in range(len(current_class_accuracy[0])):
        class_accuracy.append(get_class_accuracy([x[i] for x in current_class_accuracy] , current_total_in_class))
    sub_cvs_file.append([current_category, current_sub_category, sum(current_total_in_class)])
    sub_cvs_file[-1].extend(class_accuracy)

def sum_for_sub_categories(csv_obj):
    # print(get_category_from_class(categories, 164))
    # print(get_category_from_class(subcategories, 1))

    f = open(csv_obj.path, 'r')
    reader = csv.reader(f)
    headers = next(reader, None)

    sub_cvs_file = [['category', 'sub-category', headers[-1]]]
    sub_cvs_file[-1].extend(headers[1:-1])
    with open(csv_obj.path, 'r') as csvfile:
            if not path.exists(csv_obj.path):
                print(f"\nThe following path does not exist: {csv_obj.path}\nCode: plot.write_csv_file.py")
                sys.exit()
            
            first_row = next(reader, None)
            
            current_class_accuracy = [first_row[1:-1]]
            current_total_in_class = [int(first_row[-1])]
            current_category = get_category(first_row[0])
            current_sub_category = get_sub_category(first_row[0])

            plots = csv.reader(csvfile, delimiter=',')

            next(plots)
            next(plots)
            for row in plots:
                if current_sub_category != get_sub_category(row[0]):
                    sum_to_list(sub_cvs_file, current_category, current_class_accuracy, current_total_in_class, current_sub_category)
                    
                    if current_category != get_category(row[0]):
                        current_category = get_category(row[0])
                    
                    current_sub_category = get_sub_category(row[0])
                    current_class_accuracy = [row[1:-1]]
                    current_total_in_class = [int(row[-1])]
                    
                else:
                    current_class_accuracy.append(row[1:-1])
                    current_total_in_class.append(int(row[-1]))

            sum_to_list(sub_cvs_file, current_category, current_class_accuracy, current_total_in_class, current_sub_category)

    return sub_cvs_file

def sum_for_categories(csv_obj):
    return []
    

def split_path(csv_path):
    return_val = ''
    split_path = csv_path.split('/')[:-1]
    
    for i in range(len(split_path)):
        return_val += str(split_path[i]) + '/'
    
    return return_val

def sum_for_both(csv_path):
    csv_obj = cvs_object(csv_path)
    data = sum_for_sub_categories(csv_obj)
    csv_obj.write(data, path=f'{split_path(csv_path)}/phase2_sum_sub_cat.csv', overwrite_path=True)
    data = sum_for_categories(csv_obj.path)
    csv_obj.write(data, path='phase_two/csv_output/phase2_merged_file_categories.csv')
    
sum_for_both('phase_two/csv_output/phase2_merged_file.csv')
