from write_csv_file import cvs_object, plot

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from global_paths import get_paths


obj = cvs_object(f"{get_paths('phase_one_csv')}/old_big_boi_to_small_boi.csv")
plot([obj], lable="32")

