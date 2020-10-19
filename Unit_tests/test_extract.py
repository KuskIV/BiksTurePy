# import unittest
# import os,sys,inspect
# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir) 
# import extract as e

# class test_extract(unittest.TestCase):
#     def test_get_dataset_placements(self):
#         refrence_paths = [['Unit_tests\\Test_dataset\\00\\00000.ppm',0],
#         ['Unit_tests\Test_dataset\\00\\00001.ppm',0],
#         ['Unit_tests\Test_dataset\\00\\00002.ppm',0],
#         ['Unit_tests\Test_dataset\\00\\00003.ppm',0]]
#         refrence_amount = [4]

#         result = e.get_dataset_placements('Unit_tests\\Test_dataset')
#         exspected = (refrence_paths,refrence_amount)
        
#         self.assertTrue(result == exspected)
    
# if __name__ == '__main__':
#     unittest.main()