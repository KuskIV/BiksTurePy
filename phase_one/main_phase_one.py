
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from phase_one.experiment_one import run_experiment_one
from phase_one.find_ideal_model import get_satina_gains_model_object_list, get_satina_gains_model_norm_object_list 
# from Models.test_model import partial_accumilate_distribution, print_accumilate_distribution
# from global_paths import get_paths, get_h5_test, get_h5_train
# from general_image_func import auto_reshape_images


# def acc_dist_for_images(h5_obj:object, models:list, sizes:list, lazy_split)->None: #! NOT UPDATED TO NEW DATASET
#     accArr = np.zeros((3, 43, 2))

#     for k in range(lazy_split):
#         train_images, train_labels, _, _ = h5_obj.shuffle_and_lazyload(k, lazy_split)
#         for i in range(len(models)):
#             train_images = auto_reshape_images(sizes[i], train_images, smart_resize=True)
#             arr = partial_accumilate_distribution(train_images, train_labels, sizes[i], model=models[i])
#             for j in range(len(arr)):
#                 accArr[i][j][0] += arr[j][0]
#                 accArr[i][j][1] += arr[j][1]
    
#     for i in range(len(models)):
#         print_accumilate_distribution(accArr[i], size=sizes[i])

def ex_one(test_path, train_path, lazy_split=1, epoch_limit=100, folder_extension="experiment_one", model_paths=None):
    run_experiment_one(lazy_split, train_path, test_path, get_satina_gains_model_object_list, epochs_end=epoch_limit, folder_extension=folder_extension, model_paths=model_paths)

def ex_two_eval_norm(test_path, train_path, lazy_split=1, epoch_limit=100, folder_extension="experiment_two_eval_norm"):
    run_experiment_one(lazy_split, train_path, test_path, get_satina_gains_model_norm_object_list, epochs_end=epoch_limit, folder_extension=folder_extension)
    

# if __name__ == "__main__":
#     lazy_split = 1

#     test_path = get_h5_test()
#     train_path = get_h5_train()

#     # experiment_one(test_path, train_path)
#     experiment_two_eval_norm(test_path, train_path)
    