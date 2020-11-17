import unittest
import csv
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from phase_two.main_phase_two import find_feature_colume, group_by_feature

class Test_csv_analysis(unittest.TestCase):

    def test_find_feature_colum_true(self):
        with open('Unit_tests/csv_test_data/test_csv_analysis/test_csv_analysis.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader)
            result = find_feature_colume(header,'Original_model_45')
            self.assertEqual(result,2)

    def test_find_feature_colum_false(self):
        with open('Unit_tests/csv_test_data/test_csv_analysis/test_csv_analysis.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader)
            result = find_feature_colume(header,'Original_model_45')
            self.assertNotEqual(result,0)

    def test_find_feature_colum_nonexsistent_lable(self):
        with open('Unit_tests/csv_test_data/test_csv_analysis/test_csv_analysis.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader)
            result = find_feature_colume(header,'Original_mdel_45')
            self.assertEqual(result,None)


    def test_group_by_feature_true(self):
        with open('Unit_tests/csv_test_data/test_csv_analysis/csv_for_the_group_test.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader)
            result = group_by_feature(header,csv_reader,'filter')
            exspected = {
                'Original': [['Original', '58', '58'], ['Original', '61', '60']], 
                'night': [['night', '58', '58'], ['night', '61', '60']]
                }
            self.assertEqual(result,exspected)

    def test_group_by_feature_invalid_lable(self):
        with open('Unit_tests/csv_test_data/test_csv_analysis/csv_for_the_group_test.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            header = next(csv_reader)
            exspected = {
                'Original': [['Original', '58', '58'], ['Original', '61', '60']], 
                'night': [['night', '58', '58'], ['night', '61', '60']]
                }
            with self.assertRaises(TypeError):
                group_by_feature(header,csv_reader,'wrong')
if __name__ == '__main__':
    unittest.main()