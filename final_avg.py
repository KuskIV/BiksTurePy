import os,sys,inspect
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

def avg_longest(csv_one, csv_two, index):
    for i in range(len(csv_one)):
        for j in range(len(csv_one[i])):
            if len(csv_two) > i:
                if is_numeric(csv_one[i][j]) and is_numeric(csv_two[i][j]):
                    csv_one[i][j] = ((float(csv_one[i][j]) * index) + float(csv_two[i][j])) / (index + 1)
                elif is_numeric(csv_two[i][j]):
                    csv_one[i][j] = csv_two[i][j]
    return csv_one

def avg_two_csv(csv_one, csv_two, index):
    if len(csv_one) >= len(csv_two):
        csv_one = avg_longest(csv_one, csv_two, index)
    else:
        csv_one = avg_longest(csv_two, csv_one, index)

    return csv_one

def get_avg_csv(paths):
    return_csv = []
    
    for i in range(len(paths)):
        temp_csv = cvs_object(paths[i])
        temp_lines = temp_csv.get_lines()
        
        if len(return_csv) == 0:
            return_csv = temp_lines
        else:
            print(f"START: {paths[i]}")
            return_csv = avg_two_csv(return_csv, temp_lines, i)
            print("END")
    return return_csv

def calc_avg_from_base_and_csv(base, csv_list, output_folder):
    all_paths = walk_base(base, csv_list)
    for key, item in all_paths.items():
        path_csv = f"{output_folder}/{key}"
        avg_csv = get_avg_csv(item)
        csv_obj = cvs_object(path_csv)
        csv_obj.write(avg_csv)


def calc_avg_from_experiments(phase_one_base, phase_one_csv, phase_two_base, phase_two_csv, output_folder):
    calc_avg_from_base_and_csv(phase_one_base, phase_one_csv, output_folder)
    calc_avg_from_base_and_csv(phase_two_base, phase_two_csv, output_folder)