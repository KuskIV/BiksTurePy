import unittest
import numpy as np
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from phase_one.find_ideal_model import get_satina_gains_model_norm_object_list, get_satina_gains_model_object_list, reshape_numpy_array_of_images

class Test_find_ideal_model(unittest.TestCase):

    def test_correct_paths_baseline(self):
        expected_path = ['test_1', 'test_3', 'test_2']
        input_path = ['test_1', 'test_2', 'test_3']
        models = get_satina_gains_model_object_list(200, model_paths=input_path)
        
        self.assertEqual(expected_path, [m.path for m in models])
    
    def test_correct_paths_norm(self):
        expected_path = ['test_1', 'test_3', 'test_2']
        input_path = ['test_1', 'test_2', 'test_3']
        models = get_satina_gains_model_norm_object_list(200, model_paths=input_path)
        
        self.assertEqual(expected_path, [m.path for m in models])
    
    def test_reshape_numpy_array_of_images(self):
        img_1 = np.ones((20,20, 3))
        img_2 = np.ones((300,20, 3))
        img_3 = np.ones((20,300, 3))
        img_4 = np.ones((200,200, 3))
        img_5 = np.ones((20,20, 3))
        img_6 = np.ones((201,202, 3))
        
        expected_size = (150, 150)
        
        img_e = np.ones((expected_size[0], expected_size[1], 3))
        
        input_list = [img_1, img_2, img_3, img_4, img_5, img_6]
        expected_list = [img_e, img_e, img_e, img_e, img_e, img_e]
        
        result_list = reshape_numpy_array_of_images(input_list, expected_size)
        
        expected_sizes = [e.size for e in expected_list]
        result_sizes = [r.size for r in result_list]
        
        self.assertEqual(expected_sizes, result_sizes)