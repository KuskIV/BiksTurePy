
paths = {
            'h5':'Dataset/h5Dataset/h5set.h5py',
            'h5_train':'Dataset/h5Dataset/h5_train.h5py',
            'h5_test':'Dataset/h5Dataset/h5_test.h5py',
            'train_set':'Dataset/belgian_images/training',
            'test_set':'Dataset/belgian_images/testing',
            'model':'Models/saved_models',
            'current_model':'Models/saved_models/YEET8.h5',
            'test_img_path':'Dataset/FullJCNN2013',
            'large': 'Models/saved_models/large200.h5',
            'medium':'Models/saved_models/medium128.h5',
            'small':'Models/saved_models/small32.h5',
            'belgium':'Models/saved_models/belgium.h5',
            'phase_one_csv':'phase_one/csv_data',
            'txt_file':'labels.txt'
        }

def get_paths(key:str)->str:
    """Given a key, the value for that key is returned

    Args:
        key (str): The input key

    Returns:
        str: The corresponding value
    """
    if key in paths:
        return paths.get(key)
    else:
        print(f"ERROR: The key for path {key} is not defined.")
        return -1

def get_h5_path()->str:
    """Returns the path to the h5py file

    Returns:
        str: the path to the h5py file
    """
    return paths.get("h5")

def get_h5_train()->str:
    return paths.get('h5_train')

def get_h5_test()->str:
    return paths.get('h5_test')


def get_training_set_path()->str:
    return paths.get('train_set')

def get_data_set_path()->str:
    """Returns the path to the dataset

    Returns:
        str: the path to the dataset
    """
    return get_training_set_path()

def get_test_set_path()->str:
    return paths.get('test_set')

def get_current_model_path()->str:
    """Returns the path to the current model

    Returns:
        str: the path to the current model
    """
    return paths.get("current_model")

def get_test_img_path()->str:
    """Returns the path to the test images

    Returns:
        str: the path to the test images
    """
    return paths.get("test_img_path")

def get_model_path()->str:
    """Returns the path to the model

    Returns:
        str: the path to the model
    """
    return paths.get("model")

def get_large_model_path()->str:
    return paths.get("large")

def get_medium_model_path()->str:
    return paths.get('medium')

def get_small_model_path()->str:
    return paths.get('small')

def get_belgium_model_path()->str:
    return paths.get('belgium')

def get_test_model_paths()->tuple:
    return get_large_model_path(), get_medium_model_path(), get_small_model_path(), get_belgium_model_path()