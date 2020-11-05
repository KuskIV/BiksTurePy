
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from plot.sum_constructors.model_obj_summed_constructor import get_object as model_con
from plot.sum_main import get_class_accuracy
from plot.generalized_sum import generalized_sum

def sum_for_model(csv_obj):
    return generalized_sum(csv_obj, model_con(get_class_accuracy))
