from create_h5 import generate_h5

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from global_paths import get_h5_path, get_h5_test, get_h5_train, get_data_set_path, get_training_set_path

if __name__ == "__main__":
    train_h5_path = get_h5_train()
    test_h5_path = get_h5_test()

    train_data_path = get_data_set_path()
    test_data_path = get_training_set_path()

    generate_h5(train_h5_path, train_data_path)
    generate_h5(test_h5_path, test_data_path)


    # h5_path = get_h5_path()
    # dataset_path = get_dataset_path()
    # print(h5_path)
    # print(dataset_path)
    
    # generate_h5(h5_path, dataset_path)
    #read_h5(h5_path)