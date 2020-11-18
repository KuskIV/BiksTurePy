import csv
import sys, os
from os import path

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from plot.write_csv_file import cvs_object
from global_paths import get_paths
from error_handler import check_if_valid_path, custom_error_check

def get_indent(row):
    word = 'images'
    try:
        return row[0].index(word) + 1
    except IndexError as e:
        print(f"ERROR: {e}")
        raise IndexError
    except Exception as e:
        print(f"ERROR: {e}")
        raise Exception

def get_rows(csv_path, extension):
    check_if_valid_path(csv_path)
    
    with open(csv_path, 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            rows = list(plots)
    
    indent = get_indent(rows)
    
    for i in range(indent, len(rows[0])):
        try:
            rows[0][i] = f"{rows[0][i]}_{extension}"
        except Exception as e:
            print(f"ERROR: {e}")
            raise Exception
    
    return rows

def combine_rows(test_rows, val_rows):
    indent = get_indent(test_rows)
    
    result = []
    
    for i in range(len(test_rows)):
        result.append(test_rows[i][:indent])
    
    for i in range(len(test_rows)):
        try:
            result[i].extend(test_rows[i][indent:])#TODO one at a time pls
            result[i].extend(val_rows[i][indent:])
        except Exception as e:
            print(f"ERROR: {e}")
            raise Exception

    return result

def combine_two_summed_class_accracy(sum_test_path, sum_val_path, base_path):
    save_path = f"{base_path}/test_val_sum_class_accuracy.csv"
    test_rows = get_rows(sum_test_path, 'test')
    val_rows = get_rows(sum_val_path, 'val')
    
    rows = combine_rows(test_rows, val_rows)
    
    cvs_obj = cvs_object(save_path)
    cvs_obj.write(rows)
