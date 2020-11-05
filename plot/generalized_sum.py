import csv
import sys, os
from os import path
from write_csv_file import cvs_object
from sum_constructors.sum_constructor_object import sum_constructor

def generalized_sum(csv_obj, methods):
    f = open(csv_obj.path, 'r')
    reader = csv.reader(f)
    headers = next(reader, None)

    new_cvs_file = methods.get_headers(headers)

    with open(csv_obj.path, 'r') as csvfile:
            if not path.exists(csv_obj.path):
                print(f"\nThe following path does not exist: {csv_obj.path}\nCode: plot.write_csv_file.py")
                sys.exit()
            first_row = next(reader, None)
            current_data = methods.get_current_data(first_row)
            plots = csv.reader(csvfile, delimiter=',')

            next(plots)
            next(plots)
            for row in plots:
                if methods.category_changed(current_data, row):
                    class_accuracy = methods.calculate_class_accuracy(current_data)
                    new_cvs_file.append(methods.get_data_to_append(current_data, class_accuracy))
                    current_data = methods.get_current_data(row)
                else:
                    methods.add_to_current_data(current_data, row)
            class_accuracy = methods.calculate_class_accuracy(current_data)
            new_cvs_file.append(methods.get_data_to_append(current_data, class_accuracy))
    
    return new_cvs_file

