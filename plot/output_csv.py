import csv
from csv import reader
import pandas as pd

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from global_paths import get_paths

def get_highest_val(csv_file):
    df = pd.read_csv(csv_file)
    epoc = " "
    res = " "
    max_val_idx = df.idxmax()
    df.head(1)
    for i,j in df.iterrows():
        if i == max_val_idx[1]:
            epoc = j[0]
            res = j[2]
    return int(epoc), int(res)

def get_model_string(csv_file):
    return "model" + str(get_highest_val(csv_file)[1]) + "_" + str(get_highest_val(csv_file)[0])

def compare_csv(csv1, csv2):
    col_vals = []
    top_model = get_model_string(csv1)
    with open(csv2, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            col_vals.append(row[top_model])
    with open('ex13.csv', 'w', newline='') as csv_file:
        fieldnames = ['Class', top_model]
        print (fieldnames)
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(col_vals)):
            writer.writerow({'Class': i, top_model: col_vals[i]})





csv1 = get_paths('phase_one_csv') + "/" + "class_accuracy.csv"
csv2 = get_paths('phase_one_csv') + "/" + "model32_summed.csv"

# get_highest_val(csv2)
# get_model_string(csv2)

compare_csv(csv2,csv1)

