import os
from create_h5 import generate_h5
from adjust import AdjustETSD, run_milad
import shutil
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import Dataset.os_h5_constructor as osc
from global_paths import get_paths
from phase_two.phase_two_two import qucik_debug
from train_test_split import run_split_dataset


class InitializeDS:
    def __init__(self, new_ds_path, ):
        print('Not implemented')

    #def setup_files(path):
    #    print('Not implemented')
    
    #def initialize(path):
    #    print('Not implemented')

def delete_dest(dest_path):
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    

if __name__ == '__main__':
    path = r'Dataset/milad_gains_images'
    dest_path = r'Dataset/ETSD_Adjusted'

    delete_dest(dest_path)
    
    run_milad()
    run_split_dataset(dest_path, 0.3, path)

    train_image_path = f"{dest_path}/Training"
    test_image_path = f"{dest_path}/Testing"
    
    generate_h5(get_paths('h5_train'), train_image_path)
    generate_h5(get_paths('h5_test'), test_image_path)
    generate_h5(get_paths('h5_train_noise'), dest_path)
    generate_h5(get_paths('h5_test_noise'), dest_path)
    
    qucik_debug()
    
    

