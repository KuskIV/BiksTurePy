from create_h5 import generate_h5
from load_h5 import read_h5

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from global_paths import get_dataset_path, get_h5_path

if __name__ == "__main__":
    h5_path = get_h5_path()
    dataset_path = get_dataset_path()
    print(h5_path)
    print(dataset_path)
    generate_h5(h5_path, dataset_path)
    #read_h5(h5_path)