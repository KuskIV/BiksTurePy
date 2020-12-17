# C:\Users\madsh\OneDrive\Code\Python\BiksTurePy\phase_two\csv_output\2\experiment_big_lobster\experiment_two_big_lobster_ideal
from global_paths import get_paths
import sys, os

if __name__ == '__main__':
    phase_one_base = get_paths('phase_one_csv')
    phase_two_base = get_paths('phase_two_csv')
    
    p_one = "experiment_big_lobster/experiment_two_big_lobster_ideal"
    
    sizes_to_remove = ['49', '52']
    
    for i in range(10):
        path_to_check = os.path.join(phase_two_base, f"{i}/{p_one}")
        for path in os.listdir(path_to_check):
            if len([a for a in sizes_to_remove if a in path]) != 0:
                path_to_delete = os.path.join(path_to_check, path)
                os.remove(path_to_delete)
                print(f"{path} has been successfully deleted")