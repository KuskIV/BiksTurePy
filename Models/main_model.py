from create_model import train_model
from test_model import AccDistribution

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from global_paths import get_current_model_path

if __name__ == "__main__":
    # (train_images, train_labels, test_images, test_labels = lazyload stuff #TODO implement here
    #train_model(train_images, train_labels, test_images, test_labels, get_current_model_path(), save_model = True):
    print("this is yet to come")

    # val_images, val_labels = #TODO implement
    #AccDistribution(get_current_model_path(), val_images, val_labels)