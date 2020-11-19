import unittest

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from phase_two.sum_phase_two import merge_final_files
from plot.write_csv_file import cvs_object
import plot.sum_for_model as sum_method
from phase_one.experiment_one import sum_plot, sum_summed_plots, combine_fitdata
from phase_one.find_ideal_model import get_satina_gains_model_object_list
from plot.sum_for_model import sum_for_model, sum_for_class_accuracy, sum_summed_for_class_accuracy

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
        base_path = 'Unit_tests/csv_test_data/sum_plot_csv'
        expected_csv_path = f"{base_path}/result_model_42_test_summed.csv"
        output_csv_path = f"{base_path}/model_42_test_summed.csv"
        
        model_object_list = [get_satina_gains_model_object_list(120)[-1]]
        
        sum_plot(model_object_list, 'test', base_path)
        
        expected_csv = cvs_object(expected_csv_path)
        output_csv = cvs_object(output_csv_path)
        
        expected_rows = expected_csv.get_lines()
        output_rows = output_csv.get_lines()
        output_rows[1][1] = str(round(float(output_rows[1][1]), 2))

        self.assertEqual(expected_rows, output_rows)
    
    def test_sum_summed_plot(self):
        base_path = 'Unit_tests/csv_test_data/sum_summed_plot_csv'
        expected_csv_path = f"{base_path}/result_test_sum_summed.csv"
        output_csv_path = f"{base_path}/test_sum_summed.csv"

        model_object_list = get_satina_gains_model_object_list(120)

        sum_summed_plots(model_object_list, 'test', base_path)

        expected_csv = cvs_object(expected_csv_path)
        output_csv = cvs_object(output_csv_path)

        expected_rows = expected_csv.get_lines()
        output_rows = output_csv.get_lines()

        self.assertEqual(expected_rows, output_rows)

    def test_sum_summed_for_class_accuracy(self):
        base_path = 'Unit_tests/csv_test_data/sum_summed_for_class_accuracy_csv'
        expected_csv_path = f"{base_path}/result_test_sum_summed_class_accuracy.csv"
        output_csv_path = f"{base_path}/test_sum_summed_class_accuracy.csv"
        input_csv_path = f"{base_path}/test_sum_class_accuracy.csv"

        csv_obj = cvs_object(input_csv_path)
        data = sum_summed_for_class_accuracy(csv_obj)
        
        try:
            for i in range(1, len(data[2])):
                data[2][i] = round(data[2][i], 2) 
        except:
            print("index/cast stuff when wrong")
        
        
        csv_obj.write(data, path=output_csv_path, overwrite_path=True)

        expected_csv = cvs_object(expected_csv_path)
        output_csv = cvs_object(output_csv_path)

        expected_rows = expected_csv.get_lines()
        output_rows = output_csv.get_lines()

        self.assertEqual(expected_rows, output_rows)
    
    def test_combine_fitdata(self):
        base_path = "Unit_tests/csv_test_data/fitdata_csv"
        output_csv_path = f"{base_path}/fitdata_combined.csv"
        expected_csv_path = f"{base_path}/result_fitdata_combined.csv"
        
        model_object_list = get_satina_gains_model_object_list(120)
        
        for model_object in model_object_list:
            csv_path = f"{base_path}/{model_object.get_csv_name()}_fitdata.csv"
            csv_obj = cvs_object(csv_path)
            model_object.fit_data = csv_obj.get_lines()
        
        combine_fitdata(model_object_list, base_path)
        
        expected_csv = cvs_object(expected_csv_path)
        output_csv = cvs_object(output_csv_path)
        
        expected_rows = expected_csv.get_lines()
        output_rows = output_csv.get_lines()
        
        self.assertEqual(expected_rows, output_rows)

if __name__ == '__main__':
    unittest.main()