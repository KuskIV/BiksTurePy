import unittest
import csv
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from Noise_Generators.noise_main import premade_single_filter
from slave_main import load_level_of_fog

def load_expected_level_filters_fog()->dict:
    F1 = premade_single_filter('mod_fog0.1')
    F2 = premade_single_filter('mod_fog0.2')
    F3 = premade_single_filter('mod_fog0.3')
    F4 = premade_single_filter('mod_fog0.4')
    F5 = premade_single_filter('mod_fog0.5')
    F6 = premade_single_filter('mod_fog0.6')
    F7 = premade_single_filter('mod_fog0.7')
    F8 = premade_single_filter('mod_fog0.8')
    F9 = premade_single_filter('mod_fog0.9')
    F10 = premade_single_filter('mod_fog1.0')
    
    dict = [{'fog1':F1},{'fog2':F2},{'fog3':F3},
            {'fog4':F4},{'fog5':F5},{'fog6':F6},
            {'fog7':F7},{'fog8':F8},{'fog9':F9},
            {'fog10':F10}
        ]
    return dict

class Test_load_level_of_noise(unittest.TestCase):
    
    def test_load_level_of_noise_length(self):
        expected_filters = load_expected_level_filters_fog()
        result_filters = load_level_of_fog()
        
        self.assertTrue(len(expected_filters) == len(result_filters))
    
    def test_load_level_of_noise_keys(self):
        expected_filters = load_expected_level_filters_fog()
        result_filters = load_level_of_fog()
        
        expected_keys = [e.keys() for e in expected_filters]
        result_keys = [r.keys() for r in result_filters]
        
        self.assertTrue(expected_keys, result_keys)