import os, sys

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from plot.sum_for_model import sum_phase_2_cat_file
from plot.write_csv_file import cvs_object

def sum_merged_files_to_one_line(base_path, csv_file_name):
    for folder in os.listdir(base_path):
        folder_path = f"{base_path}/{folder}"
        
        csv_path = f"{folder_path}/{csv_file_name}"
        
        if os.path.exists(csv_path):
            return_csv_name = sum_phase_2_cat_file(folder_path, csv_file_name)
        elif not csv_path.endswith('.csv'):
            print(f"WARNING: The folder \"{folder}\" does not contain the csv file \"{csv_file_name}\". The program will continue.")
        
    return return_csv_name

def read_csv_file(csv_path):
    csv_obj = cvs_object(csv_path)
    return csv_obj.get_lines()

def sum_merged_files_to_one_file(base_path, output_csv_name):
    data_to_combine = []
    
    for folder in os.listdir(base_path):
        folder_path = f"{base_path}/{folder}"
        
        csv_path = f"{folder_path}/{output_csv_name}"
        
        if os.path.exists(csv_path):
            data_to_append = read_csv_file(csv_path)
            
            add_row_zero = True if len(data_to_combine) == 0 else False
            
            for i in range(len(data_to_append)):
                if add_row_zero:
                    data_to_combine.append(data_to_append[i])
                else:
                    data_to_combine[i].extend(data_to_append[i][1:])
        else:
            print(f"WARNING: The folder \"{folder}\" does not contain the csv file \"{output_csv_name}\". The program will continue.")
    save_path = f"{base_path}/{output_csv_name}"
    csv_to_write = cvs_object(save_path)
    csv_to_write.write(data_to_combine)

def sum_merged_files(base_path):
    if not os.path.exists(base_path):
        print(f"The input path does not exists: \"{base_path}\"")
    csv_file_name = "sum_cat.csv"
    
    output_csv_name = sum_merged_files_to_one_line(base_path, csv_file_name)
    sum_merged_files_to_one_file(base_path, output_csv_name)
    
    

if __name__ == '__main__':
    sum_merged_files('phase_two/csv_output')