import csv
import numpy as np
import matplotlib.pyplot as plt
import sys, os

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from global_paths import get_paths

class cvs_object():
    def get_file_name(self, file_name:str):
        return file_name + ".csv" if not file_name.endswith(".csv") else file_name

    def __init__(self, path:str, file_name:str):
        self.path = path + "/" + self.get_file_name(file_name)

    def write(self, data:list, file_name:str):
        with open(self.path, 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for line in data:
                filewriter.writerow(line)
    
    def plot(self, x_row=0, y_row=1):
        x = []
        y = []

        with open(self.path, 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            next(plots)
            for row in plots:
                x.append(int(row[x_row]))
                y.append(int(row[y_row]))
        plt.plot(x, y, label="Look at this graph")
        plt.xlabel("GET THIS FROM DOCUTMENT")
        plt.ylabel("GET THIS FROM DOCUTMENT")
        plt.title('this is title')
        plt.legend()
        plt.show()


# csv_obj = cvs_object(get_paths('phase_one_csv'), 'test')

# data = [['epochs', 'class', 'accury', 'resolution'], ['1', '1', '40', '32'], ['2', '3', '50', '32'], ['4', '5', '100', '200'], ['5', '7', '40', '32'], ['5', '7', '40', '32']]

# #csv_obj.write(data)
# csv_obj.plot()