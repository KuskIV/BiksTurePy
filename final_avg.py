import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from plot.write_csv_file import cvs_object

def generate_key(csv, root):
    return f"{root.split('/')[-1]}_{csv}"

def walk_base(base, csv_list):
    return_dict = {}
    for root, diretories, files in os.walk(base, topdown=False):
        for name in files:
            if name in [c[0] for c in csv_list]:
                key = generate_key(name, root)
                if not key in return_dict:
                    minus_index = [c[1] for c in csv_list if c[0] == name]
                    return_dict[key] = ([], minus_index[0])
                return_dict[key][0].append(os.path.join(root, name)) 
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
    
    # if isinstance(numb, float):
    #     return True
    # elif numb.isdigit():
    #     return True
    # else:
    #     return False
    # # return numb.isdigit()

def avg_longest(csv_one, csv_two, minus_index):
    for i in range(len(csv_one)):
        for j in range(len(csv_one[i])):
            if len(csv_two) > i:
                if is_numeric(csv_one[i][j]) and is_numeric(csv_two[i][j]):
                    # csv_one[i][j] = ((i - minus_index) * (float(csv_one[i][j])) + float(csv_two[i][j])) / ((i - minus_index) + 1)
                    t_one = csv_one[i][j]
                    csv_one[i][j] = float(csv_one[i][j]) + (float(csv_two[i][j]) - float(csv_one[i][j])) / (j - minus_index + 1)
                    print(f"{csv_one[i][j]} = {float(t_one)} + ({float(csv_two[i][j])} - {float(t_one)}) / ({j} - {minus_index} + 1)")
                elif is_numeric(csv_two[i][j]):
                    csv_one[i][j] = csv_two[i][j]
    return csv_one

def avg_two_csv(csv_one, csv_two, minus_index):
    # ensure_same_size(csv_one, csv_one)
    
    if len(csv_one) >= len(csv_two):
        csv_one = avg_longest(csv_one, csv_two, minus_index)
    else:
        csv_one = avg_longest(csv_two, csv_one, minus_index)

    return csv_one

def get_avg_csv(paths, minus_index):
    return_csv = []
    
    for i in range(len(paths[0])):
    # for p in [p[0] for p in paths]:
        temp_csv = cvs_object(paths[0][i])
        temp_lines = temp_csv.get_lines()
        
        if len(return_csv) == 0:
            return_csv = temp_lines
        else:
            return_csv = avg_two_csv(return_csv, temp_lines, paths[1])
    return return_csv

def calc_avg_from_base_and_csv(base, csv_list, output_folder):
    all_paths = walk_base(base, csv_list)
    i = 0
    for key, item in all_paths.items():
        print(i)
        i += 1
        # if i == 7:
        #     print('yeet')
        path_csv = f"{output_folder}/{key}"
        avg_csv = get_avg_csv(item, output_folder)
        csv_obj = cvs_object(path_csv)
        csv_obj.write(avg_csv)


def calc_avg_from_experiments(phase_one_base, phase_one_csv, phase_two_base, phase_two_csv, output_folder):
    calc_avg_from_base_and_csv(phase_one_base, phase_one_csv, output_folder)
    calc_avg_from_base_and_csv(phase_two_base, phase_two_csv, output_folder)