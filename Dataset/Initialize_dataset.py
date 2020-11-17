import os
from create_h5 import generate_h5
from adjust import AdjustETSD

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import Dataset.os_h5_constructor as osc
from global_paths import get_paths
from phase_two.phase_two_two import qucik_debug
from train_test_split import run_split_dataset
#from phase_two_two import qucik_debug


class InitializeDS:
    def __init__(self, new_ds_path, ):
        print('Not implemented')

    #def setup_folders(path):
    #    print('Not implemented')

    #def setup_files(path):
    #    print('Not implemented')
    
    #def initialize(path):
    #    print('Not implemented')

if __name__ == '__main__':
    path = r'Dataset/ETSDS'
    dest_path = r'Dataset/ETSDS_Final'
    a = AdjustETSD(new_ds_path=dest_path + "/",split=0.7, k=3, path_extensions = ['Training/', 'Testing/'])

    if not os.path.exists(a.new_ds_path):
         a.duplicate_ds(path)
    
    if not os.path.exists(a.german_extraction_path):
        a.extract_german_from_ETSD(path_to_original=dest_path + "/")
    
    a.re_add_german_to_adjusted_ds()
    
    width_threshold = 10
    height_threshold = 10

    a.trim_imgs_according_to_predicate(lambda pil_image: pil_image.width < width_threshold or pil_image.height < height_threshold)

    run_split_dataset(dest_path, 0.3)

    generate_h5(get_paths('h5_train'), dest_path)
    generate_h5(get_paths('h5_test'), dest_path)
    generate_h5(get_paths('h5_train_noise'), dest_path)
    generate_h5(get_paths('h5_test_noise'), dest_path)
    
    qucik_debug
    
    

