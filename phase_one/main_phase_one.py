
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from phase_one.experiment_one import run_experiment_one
from phase_one.find_ideal_model import get_satina_gains_model_object_list, get_satina_gains_model_norm_object_list 

def ex_one(test_path:str, train_path:str, lazy_split:int=1, epoch_limit:int=100, folder_extension:str="experiment_one", model_paths:list=None)->None:
    """Performs experiment one. This will train and save at default three models, in the folder 'folder_extension'

    Args:
        test_path (str): path to the test h5 file
        train_path (str): path to the train h5 file
        lazy_split (int, optional): how many splits the train and test images are split into. Defaults to 1.
        epoch_limit (int, optional): the upperlimit of how many epochs each model can be trained for. Defaults to 100.
        folder_extension (str, optional): the name of the folder where the csv data from the models will be saved. Defaults to "experiment_one".
        model_paths (list, optional): if the models should be saved in another path, this will be the list of pats. Defaults to None.
    """
    run_experiment_one(lazy_split, train_path, test_path, get_satina_gains_model_object_list, epochs_end=epoch_limit, folder_extension=folder_extension, model_paths=model_paths)

def ex_two_eval_norm(test_path:str, train_path:str, lazy_split:int=1, epoch_limit:int=100, folder_extension:str="experiment_two_eval_norm"):
    """Trains at default three models with a normalization layer implemented, in a similar fashion as was done in experiment one

    Args:
        test_path (str): path to the test h5 file
        train_path (str): path to the train h5 file
        lazy_split (int, optional): how many splits the train an test datasets shouold be split into. Defaults to 1.
        epoch_limit (int, optional): the upper limit of how many epochs the models can be trained for. Defaults to 100.
        folder_extension (str, optional): the folder where the csv files will be saved from the experiment. Defaults to "experiment_two_eval_norm".
    """
    run_experiment_one(lazy_split, train_path, test_path, get_satina_gains_model_norm_object_list, epochs_end=epoch_limit, folder_extension=folder_extension)
