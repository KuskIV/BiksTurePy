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

class cvs_object():
    def __init__(self, path:str, x_row=0, y_row=1, label="UNKNOWN"):
        self.path = path
        self.x_row = x_row
        self.y_row = y_row
        self.label = label

    def write(self, data:list):
        with open(self.path, 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for line in data:
                filewriter.writerow(line)
    
def plot(cvs_list:object, title="_", lable="_")->None:
    
    for cvs in cvs_list:
        x = []
        y = []

        x_label = ""
        y_label = ""

        with open(cvs.path, 'r') as csvfile:
            if not path.exists(cvs.path):
                print(f"\nThe following path does not exist: {cvs.path}\nCode: plot.write_csv_file.py")
                sys.exit()

            plots = csv.reader(csvfile, delimiter=',')
            
            if not x_label == "":
                next(plots)
            for row in plots:
                if len(row) >= int(cvs.x_row) and len(row) >= int(cvs.y_row):
                    if x_label == "":
                        x_label = row[cvs.x_row]
                        y_label = row[cvs.y_row]
                    else:
                        x.append(int(row[cvs.x_row]))
                        y.append(int(row[cvs.y_row]))

        plt.plot(x, y, label=f"Resolution: {cvs.label}")
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


csv_obj01 = cvs_object(get_paths('phase_one_csv') + "/" + 'test01.cvs', label="200")
csv_obj02 = cvs_object(get_paths('phase_one_csv') + "/" + 'test02.cvs')

data01 = [['epochs', 'class', 'accury', 'resolution'], ['1', '1', '40', '32'], ['2', '3', '50', '32'], ['4', '5', '100', '200'], ['5', '7', '40', '32'], ['5', '7', '40', '32']]
data02 = [['epochs', 'class', 'accury', 'resolution'], ['1', '2', '40', '32'], ['2', '4', '50', '32'], ['4', '6', '100', '200'], ['5', '8', '40', '32'], ['5', '9', '40', '32']]

csv_obj01.write(data01)
csv_obj02.write(data02)

plot([csv_obj01, csv_obj02], title="This is the title", lable="This is the lable")