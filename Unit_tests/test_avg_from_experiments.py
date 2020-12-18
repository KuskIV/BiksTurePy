import unittest
import csv
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from final_avg import calc_avg_from_base_and_csv
from plot.write_csv_file import cvs_object

def construct_dict(folder, extensions):
    temp_dict = {}
    for p in os.listdir(folder):
        if p.endswith('.csv') and extensions in p:
            temp_path = f"{folder}/{p}"
            temp_csv = cvs_object(temp_path)
            temp_dict[p] = temp_csv.get_lines()
    return temp_dict

def compare_csv(r_line, e_line):
    return r_line == e_line

class Test_csv_analysis(unittest.TestCase):
    
    def test_calc_avg_from_base_and_csv(self):
        base = 'Unit_tests/csv_test_data/final_avg_csv/input'
        csv_list = ['aaa.csv', 'bbb.csv', 'ccc.csv']
        output_folder = 'Unit_tests/csv_test_data/final_avg_csv/ouput/result'
        expected_folder = 'Unit_tests/csv_test_data/final_avg_csv/ouput/expected'
        extension = 'avg'
        result_bool = True
        
        calc_avg_from_base_and_csv(base, csv_list, output_folder, calc_deviant=False)
        
        expected_csv = construct_dict(expected_folder, extension)
        result_csv = construct_dict(output_folder, extension)
        
        for key in expected_csv.keys():
            result_bool = compare_csv(expected_csv[key], result_csv[key])
            
            if not result_bool:
                break
        
        self.assertTrue(result_bool)
        
    def test_calc_deviance_from_base_and_csv(self):
        base = 'Unit_tests/csv_test_data/final_avg_csv/input'
        csv_list = ['aaa.csv']
        output_folder = 'Unit_tests/csv_test_data/final_avg_csv/ouput/result'
        expected_folder = 'Unit_tests/csv_test_data/final_avg_csv/ouput/expected'
        extension = 'deviant'
        result_bool = True
        
        calc_avg_from_base_and_csv(base, csv_list, output_folder)
        
        expected_csv = construct_dict(expected_folder, extension)
        result_csv = construct_dict(output_folder, extension)
        
        for key in expected_csv.keys():
            result_bool = compare_csv(expected_csv[key], result_csv[key])
            
            if not result_bool:
                break
        
        self.assertTrue(result_bool)