from write_csv_file import cvs_object, plot
import os.path
from os import path
import sys, os
import csv

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from global_paths import get_paths

def get_class_accuracy(current_class_accuracy, current_total_in_class):
    total = 0
    right = 0

    for i in range(len(current_class_accuracy)):
        total += float(current_total_in_class[i])
        right += float(current_total_in_class[i]) * (float(current_class_accuracy[i]) / 100)

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
                    current_class_accuracy.append(row[3])
                    current_total_in_class.append(row[4])
                    current_resolution = row[1]
                else:
                    current_class_accuracy.append(row[3])
                    current_total_in_class.append(row[4])
            class_accuracy = get_class_accuracy(current_class_accuracy, current_total_in_class)
            new_cvs_file.append([current_epoc, class_accuracy, current_resolution])
    return new_cvs_file

obj = cvs_object(f"{get_paths('phase_one_csv')}/big_boi.csv")
data = sum_csv(obj)

new_obj = obj.write(data, path=f"{get_paths('phase_one_csv')}/big_boi_to_small_boi.csv")
plot([new_obj], lable="32")


                
