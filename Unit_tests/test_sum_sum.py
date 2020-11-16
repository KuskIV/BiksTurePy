import unittest

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from phase_two.sum_phase_two import merge_final_files
from plot.write_csv_file import cvs_object

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
        

if __name__ == '__main__':
    unittest.main()