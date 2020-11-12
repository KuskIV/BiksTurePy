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

def get_x_row(row:list)->str:
    return row[0]

def get_y_row(row:list)->str:
    return row[1]

def set_x_y_lables(headers:list)->tuple:
    return headers[0], headers[1]


class cvs_object():
    def __init__(self, path:str, get_x_row=get_x_row, get_y_row=get_y_row, label="UNKNOWN", set_x_y_lables=set_x_y_lables):
        self.path = path
        self.get_x_row = get_x_row
        self.get_y_row = get_y_row
        self.label = label
        self.set_x_y_lables = set_x_y_lables

    def write(self, data:list, path:str="", overwrite_path:bool=False, return_object:bool=False)->None:
        
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
        
    def get_lines(self):
        if not os.path.exists(self.path):
            print(f"ERROR: The csv file \"{self.path}\" does not exists. The program will now exit.")
            sys.exit()
        
        with open(self.path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)
        return data

    
def plot(cvs_list:object, title:str="look at this graph", lable:str="_")->None:
    
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
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()