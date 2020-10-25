import csv
import numpy as np
import matplotlib.pyplot as plt
import os.path
from os import path
import sys, os

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from global_paths import get_paths

def get_x_row(row):
    return row[0]

def get_y_row(row):
    return row[1]

def set_x_y_lables(headers):
    return headers[0], headers[1]


class cvs_object():
    def __init__(self, path:str, get_x_row=get_x_row, get_y_row=get_y_row, label="UNKNOWN", set_x_y_lables=set_x_y_lables):
        self.path = path
        self.get_x_row = get_x_row
        self.get_y_row = get_y_row
        self.label = label
        self.set_x_y_lables = set_x_y_lables

    def write(self, data:list, path="", overwrite_path=False, return_object=False):
        
        path = self.path if path == "" else path

        if overwrite_path:
            self.path = path

        with open(path, 'w', newline="") as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for line in data:
                filewriter.writerow(line)

        if return_object:
            return cvs_object(path)
        

    
def plot(cvs_list:object, title="_", lable="_")->None:
    
    f = open(cvs_list[0].path, 'r')
    reader = csv.reader(f)
    headers = next(reader, None)

    x_label, y_label = cvs_list[0].set_x_y_lables(headers)

    for cvs in cvs_list:
        x = []
        y = []

        with open(cvs.path, 'r') as csvfile:
            if not path.exists(cvs.path):
                print(f"\nThe following path does not exist: {cvs.path}\nCode: plot.write_csv_file.py")
                sys.exit()

            plots = csv.reader(csvfile, delimiter=',')
            
            next(plots)
            for row in plots:
                x.append(float(cvs.get_x_row(row)))
                y.append(float(cvs.get_y_row(row)))

        plt.plot(x, y, label=f"Resolution: {cvs.label}")
    # plt.ylim(80, 90) #TODO Calculate this dynamicly
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


# csv_obj01 = cvs_object(get_paths('phase_one_csv') + "/" + 'test01.cvs', label="200")
# csv_obj02 = cvs_object(get_paths('phase_one_csv') + "/" + 'test02.cvs')

# data01 = [['epochs', 'class', 'accury', 'resolution'], ['1', '1', '40', '32'], ['2', '3', '50', '32'], ['4', '5', '100', '200'], ['5', '7', '40', '32'], ['5', '7', '40', '32']]
# data02 = [['epochs', 'class', 'accury', 'resolution'], ['1', '2', '40', '32'], ['2', '4', '50', '32'], ['4', '6', '100', '200'], ['5', '8', '40', '32'], ['5', '9', '40', '32']]

# csv_obj01.write(data01)
# csv_obj02.write(data02)

# plot([csv_obj01, csv_obj02], title="This is the title", lable="This is the lable")

# def get_y_colum(row):
#     #total_in_class = row[4]
#     #class_accuicy
#     return row[1]

# def set_x_y_lables_2(headers):
#     return headers[0], headers[1]

# obj = cvs_object(f"{get_paths('phase_one_csv')}/big_boi.csv", get_y_row=get_y_colum, set_x_y_lables=set_x_y_lables_2)
# plot([obj], title="This is a big table")