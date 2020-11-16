import unittest

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from phase_two.sum_phase_two import merge_final_files
from plot.write_csv_file import cvs_object
import plot.sum_for_model as sum_method
from phase_one.experiment_one import sum_plot, sum_summed_plots
from phase_one.find_ideal_model import get_satina_gains_model_object_list

class Test_sum_sum(unittest.TestCase):
    def test_merge_final_files(self):
        output_csv_name = "final_sum_sum_sum_summed.csv"
        base_path = "Unit_tests/csv_test_data/merge_final_files_csv"
        
        merge_final_files(base_path, output_csv_name)
        
        expected_csv = cvs_object(f"{base_path}/result_{output_csv_name}")
        output_csv = cvs_object(f"{base_path}/{output_csv_name}")
        
        expected_rows = expected_csv.get_lines()
        output_lines = output_csv.get_lines()
        
        self.assertEqual(expected_rows, output_lines)
    
    def test_sum_plot(self):
        base_path = 'Unit_tests/csv_test_data/sum_plot'
        expected_csv_path = f"{base_path}/result_model_24_test.csv"
        output_csv_path = f"{base_path}/model_24_test.csv"
        
        model_object_list = [get_satina_gains_model_object_list()[-1]]
        
        sum_plot(model_object_list, 'test', base_path)
        
        expected_csv = cvs_object(expected_csv_path)
        output_csv = cvs_object(output_csv_path)
        
        expected_rows = expected_csv.get_lines()
        output_rows = output_csv.get_lines()

        self.assertEqual(expected_rows, output_rows)
    
    def test_sum_summed_plot(self):
        base_path = 'Unit_tests/csv_test_data/sum_plot'
        expected_csv_path = f"{base_path}/result_model_24_test_summed.csv"
        output_csv_path = f"{base_path}/model_24_test_summed.csv"

        model_object_list = get_satina_gains_model_object_list()

        sum_summed_plots(model_object_list, 'test', base_path)

        expected_csv = cvs_object(expected_csv_path)
        output_csv = cvs_object(output_csv_path)

        expected_rows = expected_csv.get_lines()
        output_rows = output_csv.get_lines()

        self.assertEqual(expected_rows, output_rows)

if __name__ == '__main__':
    unittest.main()