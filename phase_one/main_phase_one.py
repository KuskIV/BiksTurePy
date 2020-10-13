from find_ideal_model import get_processed_models, train_and_eval_models_for_size

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from general_image_func import get_class_names # This is not an error
from Dataset.load_h5 import lazyload_h5, get_h5
from global_paths import get_h5_path

def find_ideal_model():
    img_dataset = [] # list of all images in reshaped numpy array
    img_labels = [] # labels for all images in correct order
    images_per_class = [] # list, where each entry represents the number of ppm images for that classification class
    class_names = [] # classification text for labels

    class_names = get_class_names()

    image_sizes = [(32, 32), (128, 128), (200, 200)]

    #img_dataset, img_labels, images_per_class = get_data(fixed_size = (32, 32), padded_images = False, smart_resize = True)
    
    # Training and test split, 70 and 30%
    split = 2
    h5 = get_h5(get_h5_path())

    for j in range(split):
        # generate models
        train_images, train_labels, test_images, test_labels = lazyload_h5(h5, j, split)
        #models = get_processed_models()

        # zip together with its size
        #model_and_size = list(zip(models, image_sizes))

        # train models
        #for i in range(len(model_and_size)):
        #    print(f"Training model {i} / {len(model_and_size) - 1} for time {j} / {split - 1}")
        #    train_and_eval_models_for_size(models, model_and_size[i][1], model_and_size[i][0], i, train_images, train_labels, test_images, test_labels)

if __name__ == "__main__":
    find_ideal_model()