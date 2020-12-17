import os,sys,inspect
import copy
import math
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from plot.write_csv_file import cvs_object

def generate_key(csv, root):
    root = root.replace('\\', '/')
    return f"{root.split('/')[-1]}_{csv}"

def walk_base(base, csv_list):
    return_dict = {}
    for root, diretories, files in os.walk(base, topdown=False):
        for name in files:
            if name in csv_list:
                key = generate_key(name, root)
                if not key in return_dict:
                    return_dict[key] = []
                return_dict[key].append(os.path.join(root, name).replace('\\', '/')) 
    return return_dict

def ensure_same_size(csv_one, csv_two):
    if len(csv_one) != len(csv_two):
        raise IndexError
    
    for i in range(len(csv_one)):
        if len(csv_one[i]) != len(csv_two[i]):
            raise IndexError

def is_numeric(numb):
    try:
        _ = float(numb)
    except Exception:
        return False
    else:
        return True

def avg_calc(csv_one, csv_two, index):
    return ((float(csv_one) * index) + float(csv_two)) / (index + 1)

def deviant_calc_first_time(csv_one, csv_two, avg):
    return (float(csv_one) - float(avg)) ** 2

def deviant_calc(csv_one, csv_two, avg):
    return float(csv_one) + ((float(csv_two) - float(avg)) ** 2)

def divide_calc(deviant_csv, length):
    return math.sqrt(float(deviant_csv) / float(length))

def divide_csv(csv_arr, length):
    for i in range(len(csv_arr)):
        for j in range(len(csv_arr[i])):
            if is_numeric(csv_arr[i][j]):
                csv_arr[i][j] = divide_calc(csv_arr[i][j], length)
    return csv_arr

def process_longest(csv_one, csv_two, avg_csv, index, calc_method):
    for i in range(len(csv_one)):
        for j in range(len(csv_one[i])):
            if len(csv_two) > i:
                if is_numeric(csv_one[i][j]) and is_numeric(csv_two[i][j]):
                    if avg_csv == None:
                        csv_one[i][j] = calc_method(csv_one[i][j], csv_two[i][j], index)
                    else:
                        csv_one[i][j] = calc_method(csv_one[i][j], csv_two[i][j], avg_csv[i][j])
                elif is_numeric(csv_two[i][j]):
                    csv_one[i][j] = csv_two[i][j]
    return csv_one

def process_two_csv(csv_one, csv_two, index, calc_method, avg_csv=None):
    if len(csv_one) >= len(csv_two):
        csv_one = process_longest(csv_one, csv_two, avg_csv, index, calc_method)
    else:
        csv_one = process_longest(csv_two, csv_one, avg_csv, index, calc_method)

    return csv_one

def get_processed_csv(paths, calc_deviant, length):
    avg_csv = []
    deviant_csv = []
    
    for i in range(len(paths)):
        temp_csv = cvs_object(paths[i])
        temp_lines = temp_csv.get_lines()
        
        if len(avg_csv) == 0:
            avg_csv = temp_lines.copy()
        else:
            avg_csv = process_two_csv(copy.deepcopy(avg_csv), copy.deepcopy(temp_lines), i, avg_calc)

    for i in range(len(paths)):
        temp_csv = cvs_object(paths[i])
        temp_lines = temp_csv.get_lines()
        
        if calc_deviant:
            if len(deviant_csv) == 0:
                deviant_csv = process_two_csv(copy.deepcopy(temp_lines), copy.deepcopy(temp_lines), i, deviant_calc_first_time, avg_csv=avg_csv)
            else:
                deviant_csv = process_two_csv(copy.deepcopy(deviant_csv), copy.deepcopy(temp_lines), i, deviant_calc, avg_csv=avg_csv)

    if calc_deviant:
        deviant_csv = divide_csv(deviant_csv, length)
    
    return avg_csv, deviant_csv

def get_avg_path(base_path):
    return f"{base_path.split('.')[0]}_avg.csv"

def get_deviation_path(base_path):
    return f"{base_path.split('.')[0]}_deviant.csv"

def calc_avg_from_base_and_csv(base, csv_list, output_folder, calc_deviant=True):
    all_paths = walk_base(base, csv_list)
    for key, item in all_paths.items():
        path_csv = f"{output_folder}/{key}"
        avg_csv, deviant_csv = get_processed_csv(item, calc_deviant, len(item))
        
        avg_csv_obj = cvs_object(get_avg_path(path_csv))
        avg_csv_obj.write(avg_csv)
        
        if calc_deviant:
            deviant_csv_obj = cvs_object(get_deviation_path(path_csv))
            deviant_csv_obj.write(deviant_csv)


def calc_avg_from_experiments(phase_one_base, phase_one_csv, phase_two_base, phase_two_csv, output_folder, calc_deviant=None):
    phase_one_calc_deviant = False if calc_deviant == None else calc_deviant
    phase_two_calc_deviant = True if calc_deviant == None else calc_deviant
    
    calc_avg_from_base_and_csv(phase_one_base, phase_one_csv, output_folder, calc_deviant=phase_one_calc_deviant)
    calc_avg_from_base_and_csv(phase_two_base, phase_two_csv, output_folder, calc_deviant=phase_two_calc_deviant)