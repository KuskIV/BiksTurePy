
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from global_paths import get_paths
from plot.write_csv_file import cvs_object

from plot.sum_constructors.summed_class_accuracy_constructor import get_object as summed_acc_con
from plot.sum_constructors.class_accuracy_constructor import get_object as acc_con
from plot.sum_constructors.model_obj_summed_constructor import get_object as model_con
from plot.sum_main import get_class_accuracy, get_sub_category, get_category
from plot.generalized_sum import generalized_sum
from plot.phase2_merged_file_sum import sum_merged_file

def sum_for_model(csv_obj):
    return generalized_sum(csv_obj, model_con(get_class_accuracy))

def sum_for_class_accuracy(csv_obj):
    return generalized_sum(csv_obj, acc_con(get_sub_category, get_category, get_class_accuracy))

def sum_summed_for_class_accuracy(csv_obj):
    return generalized_sum(csv_obj, summed_acc_con(get_category, get_class_accuracy))

def sum_phase_2_files():
    sum_merged_file(get_category, get_sub_category, get_class_accuracy)



