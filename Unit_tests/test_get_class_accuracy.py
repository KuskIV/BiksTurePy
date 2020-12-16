import unittest
import csv
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from plot.sum_main import get_class_accuracy

class Test_get_class_accuracy(unittest.TestCase):
    def test_get_class_accuracy(self):
        pass
