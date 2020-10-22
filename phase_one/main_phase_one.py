from find_ideal_model import get_processed_models, train_and_eval_models_for_size
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
from tqdm import trange
import os
import sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from plot.write_csv_file import cvs_object, plot
from Dataset.load_h5 import h5_object
from Models.create_model import store_model
from Models.test_model import partial_accumilate_distribution, print_accumilate_distribution, make_prediction
from global_paths import get_test_model_paths, get_paths, get_h5_test, get_h5_train
from general_image_func import auto_reshape_images, convert_numpy_image_to_image, load_images_from_folders

def acc_dist_for_images(h5_obj:object, models:list, sizes:list, lazy_split)->None:
    accArr = np.zeros((3, 43, 2))

    for k in range(lazy_split):
        train_images, train_labels, _, _ = h5_obj.shuffle_and_lazyload(k, lazy_split)
        for i in range(len(models)):
            train_images = auto_reshape_images(sizes[i], train_images, smart_resize=True)
            arr = partial_accumilate_distribution(train_images, train_labels, sizes[i], model=models[i])
            for j in range(len(arr)):
                accArr[i][j][0] += arr[j][0]
                accArr[i][j][1] += arr[j][1]
    
    for i in range(len(models)):
        print_accumilate_distribution(accArr[i], size=sizes[i])


def find_ideal_model(h5_obj:object, image_sizes, models, epochs=10, lazy_split=10)->None:
    """finds the ideal model

    Args:
        h5_obj (object): h5 object
    """

    train_images = []
    test_images = []
    for j in range(lazy_split):
        
        # generate models
        train_images, train_labels, test_images, test_labels = h5_obj.shuffle_and_lazyload(j, lazy_split)
        
        print(f"Images in train_set: {len(train_images)} ({len(train_images) == len(train_labels)}), Images in val_set: {len(test_images)} ({len(test_images) == len(test_labels)})")
        print(f"This version will split the dataset in {lazy_split} sizes.")
        
        # zip together with its size
        model_and_size = list(zip(models, image_sizes))

        # train models
        for i in range(len(model_and_size)):
            print(f"Training model {i + 1} / {len(model_and_size) } for epoch {j + 1} / {lazy_split}")
            train_and_eval_models_for_size(models, model_and_size[i][1], model_and_size[i][0], i, train_images, train_labels, test_images, test_labels, epochs)
    
    large_model_path, medium_model_path, small_model_path = get_test_model_paths()

    #store_model(models[0], large_model_path)
    #store_model(models[1], medium_model_path)
    store_model(models[0], small_model_path) # SHOULD BE INDEX 2



if __name__ == "__main__":
    lazy_split = 1
    dataset_split = 0.8

    image_sizes = [(32, 32)]

    test_path = get_h5_test()
    train_path = get_h5_train()

    #image_dataset = auto_reshape_images(image_sizes[0], image_dataset)

    h5_obj = h5_object(train_path, training_split=dataset_split)

    models = [get_processed_models(output_layer_size=h5_obj.class_in_h5)[2]] # SHOULD NOT BE A LIST


    find_ideal_model(h5_obj, image_sizes, models, epochs=50, lazy_split=lazy_split)


    # path = "/home/biks/Desktop"
    # test_path = "/imagesForMads"

    # class_names = open(get_paths("txt_file"), 'r').readlines()

    # image_sizes = [(32, 32)]
    # models = [get_processed_models()[2]] # SHOULD NOT BE A LIST

    # image_dataset, lable_dataset = load_images_from_folders(path + test_path)
    # label_dict = {}

    # big_boi_list = [['Epochs', 'Resolution', 'Class', 'Class_Acuracy', 'Total_in_Class']]
    # small_boi_list = [['Epochs', 'Accuracy', 'Resolution']]
    
    # for e in range(3, 31):
    #     print(f"\n----------------\nRun {e} / {15}\n----------------\n")
    #     find_ideal_model(h5_obj, image_sizes, models, epochs=e)
        
    #     large_model_path, medium_model_path, small_model_path = get_test_model_paths()
        
    #     #large_model = tf.keras.models.load_model(large_model_path)
    #     #medium_model = tf.keras.models.load_model(medium_model_path)
    #     small_model = tf.keras.models.load_model(small_model_path)

    #     # loaded_models = [large_model, medium_model, small_model]
    #     loaded_models = [small_model]
        
    #     print(f"\n------------------------\nTraining done. Now evaluation will be made, using {e} epochs.\n\n")

    #     for i in range(len(loaded_models)):
    #         for lable in lable_dataset:
    #             label_dict[lable] = [0,0]

    #         done = len(image_dataset)
    #         classes = trange(len(image_dataset), desc='Image stuff', leave=True)
    #         image_dataset = auto_reshape_images(image_sizes[i], image_dataset)
    #         for j in range(len(image_dataset)):
    #             classes.set_description(f"Image {j + 1} / {done}")
    #             classes.refresh()

    #             right = 0
    #             wrong = 0
                
    #             prediction = make_prediction(loaded_models[i], image_dataset[j].copy())
    #             predicted_label = np.argmax(prediction)
    #             softmaxed = tf.keras.activations.softmax(prediction)
    #             img = convert_numpy_image_to_image(image_dataset[j])

    #             if int(predicted_label) == int(lable_dataset[j]):
    #                 right += 1
    #                 label_dict[lable_dataset[j]][1] += 1
    #                 #img.save(f"{path}/right/{i}_{predicted_label}_{int(100*np.max(softmaxed))}_{ephocs}.jpg")
    #             else:
    #                 wrong += 1
    #                 label_dict[lable_dataset[j]][0] += 1
    #                 #img.save(f"{path}/wrong/{i}_{lable_dataset[i]}_{predicted_label}_{int(100*np.max(softmaxed))}_{ephocs}.jpg")

    #         percent = (right / (wrong + right)) * 100

    #         small_boi_list.append([e, percent, image_sizes[i]])

    #         print(f"Right: {right}, wrong: {wrong}, percent correct: {percent}")
            
    #         for key in label_dict.keys():
    #             class_name = str(key).zfill(2)
    #             right_name = str(label_dict[key][1]).rjust(4, ' ')
    #             wrong_name = str(label_dict[key][0]).rjust(4, ' ')
    #             class_size = label_dict[key][0]+label_dict[key][1]
    #             class_percent = (label_dict[key][1]/class_size)*100

    #             big_boi_list.append([e, image_sizes[i][0], class_name, class_percent, class_size])


    #             print(f"class: {class_name} | right: {right_name} | wrong: {wrong_name} | procent: {round(class_percent, 2)}")

    #         #cvs_obj = cvs_object(f"get_paths('phase_one_csv')/big_boi" , x_row=0, y_row=1, label=str(image_sizes[i]))
    #         #cvs_obj.write([])

    
    # cvs_obj = cvs_object(f"{get_paths('phase_one_csv')}/big_boi.csv" , x_row=0, y_row=1)
    # cvs_obj.write(big_boi_list)
    # plot([cvs_obj])

    # big_boi_list = [['Epochs', 'Resolution', 'Class', 'Class_Acuracy', 'Total_in_Class']]

    
    #acc_dist_for_images(h5_obj, [large_model, medium_model, small_model], [(200, 200), (128, 128), (32, 32)], 20)
    # # acc_dist_for_images(h5_obj, [large_model_path, medium_model_path, small_model_path], 10)
    
    # # This was a table generator for roni
    # h5_obj.print_class_data()