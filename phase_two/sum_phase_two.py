import os, sys

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from plot.sum_for_model import sum_phase_2_cat_file
from plot.write_csv_file import cvs_object
from error_handler import check_if_valid_path, custom_error_check

def append_extension_to_header(header, extension):
    return f"{header}_{extension}" if extension not in header else header

def append_to_headers(header_extension:str, csv_path:str)->None:
    """add an extension to all models in the headers before combining them

    Args:
        header_extension (str): the extension to add
        csv_path (str): the path to the csv file to edit
    """
    csv_obj = cvs_object(csv_path)
    rows = csv_obj.get_lines()
    new_rows = []
    new_rows = [append_extension_to_header(x, header_extension) for x in rows[0][2:]]
    
    del rows[0][2:]
    rows[0].extend(new_rows)
    csv_obj.write(rows)

def sum_merged_files_to_one_line(base_path:str, csv_file_name:str)->str:
    """Iterates through all folders in the base path, and if the csv_file_name exists, they are summed into one line

    Args:
        base_path (str): the base path to iterate through
        csv_file_name (str): the csv file to look for

    Returns:
        str: the name of the file the csv files are combined into
    """
    for folder in os.listdir(base_path):
        folder_path = f"{base_path}/{folder}"
        
        csv_path = f"{folder_path}/{csv_file_name}"
        
        if os.path.exists(csv_path):
            if '_' in folder:
                header_extension = folder.split('_')[-1]
                if len(header_extension) == 0:
                    print(f"ERROR: the header extension is empty (sum_phase_two.py)")
            else:
                print(f"ERROR: The folder \"{folder}\" is not in a valid. Should contain a '_' at the end, followed by what experiment it is (sum_phase_two.py)")
                sys.exit()
            append_to_headers(header_extension, csv_path)
            return_csv_name = sum_phase_2_cat_file(folder_path, csv_file_name)
        elif not csv_path.endswith('.csv'):
            print(f"WARNING: The folder \"{folder}\" does not contain the csv file \"{csv_file_name}\". The program will continue.")
        
    return return_csv_name

def read_csv_file(csv_path:str)->list:
    """reas a csv file and returns all the lines as a list

    Args:
        csv_path (str): the csv file to open

    Returns:
        list: the list of rows
    """
    csv_obj = cvs_object(csv_path)
    return csv_obj.get_lines()

def sum_merged_files_to_one_file(base_path, output_csv_name):
    """iterates through all folders in the base path, and combines 

    Args:
        base_path ([type]): [description]
        output_csv_name ([type]): [description]
    """
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
        elif not csv_path.endswith('.csv'):
            print(f"WARNING: The folder \"{folder}\" does not contain the csv file \"{output_csv_name}\". The program will continue.")
    save_path = f"{base_path}/{output_csv_name}"
    csv_to_write = cvs_object(save_path)
    csv_to_write.write(data_to_combine)

def verify_cell(cell_data):
    return len(cell_data) == 4

def dissasemble_cell(cell):
    cell_data = cell.split('_')
    custom_error_check(verify_cell(cell_data), f'The cell {cell} is not in the correct syntax. Should contain four underscores')
    return cell_data[0], f"{cell_data[3]}_{cell_data[2]}"

def data_index_exists(data_to_append):
    return len(data_to_append) == 2

def second_index_exists(data_to_append, i):
    return len(data_to_append) >= i

def convert_to_list(data_to_combine):
    return_list = []
    for key, item in data_to_combine.items():
        return_list.append([key])
        return_list[-1].extend(item)
    return return_list

def verify_dimensions(data_list):
    length = 0
    
    for row in data_list:
        if length == 0:
            length = len(row)
        elif len(row) != length:
            return False
            
    return True

def merge_combined_files(data_to_combine, base_path, output_csv_name):
    data_list = convert_to_list(data_to_combine)
    
    custom_error_check(verify_dimensions(data_list), 'data_list has the wrong dimensions')
    
    data_to_save = []
    
    for i in range(len(data_list)):
        for j in range(len(data_list[0])):
            if len(data_to_save) <= j:
                data_to_save.append([])
            
            data_to_save[j].append(data_list[i][j])
            # data_to_save[-1].extend(data_list[j][i])
    
    save_path = f"{base_path}/{output_csv_name}"
    csv_to_write = cvs_object(save_path)
    csv_to_write.write(data_to_save)

def merge_final_files(base_path, output_csv_name):
    """iterates through all folders in the base path, and combines 

    Args:
        base_path ([type]): [description]
        output_csv_name ([type]): [description]
    """
    data_to_combine = {'filter':[]}
    
    for folder in os.listdir(base_path):
        folder_path = f"{base_path}/{folder}"
        
        csv_path = f"{folder_path}/{output_csv_name}"
        if os.path.exists(csv_path):
            data_to_append = read_csv_file(csv_path)
            custom_error_check(data_index_exists(data_to_append), f"The list data_to_append should have length 2, but is {len(data_to_append)}.")
            for i in range(1, len(data_to_append[0])):
                noise, experiment = dissasemble_cell(data_to_append[0][i])
                if noise not in data_to_combine['filter']:
                    data_to_combine['filter'].append(noise)
                
                if experiment not in data_to_combine:
                    data_to_combine[experiment] = []
                
                custom_error_check(second_index_exists(data_to_append[1], i), f"Index {i} is being accesed on list data_to_append, but this is out of range")
                data_to_combine[experiment].append(data_to_append[1][i])
                
        elif not csv_path.endswith('.csv'):
            print(f"WARNING: The folder \"{folder}\" does not contain the csv file \"{output_csv_name}\". The program will continue.")
    
    merge_combined_files(data_to_combine, base_path, output_csv_name)

def sum_merged_files(base_path):
    if not os.path.exists(base_path):
        print(f"The input path does not exists: \"{base_path}\"")
    csv_file_name = "sum_cat.csv"
    
    output_csv_name = "final_sum_sum_sum_summed.csv" #sum_merged_files_to_one_line(base_path, csv_file_name)
    # sum_merged_files_to_one_file(base_path, output_csv_name)
    merge_final_files(base_path, output_csv_name)

if __name__ == '__main__':
    sum_merged_files('phase_two/csv_output')