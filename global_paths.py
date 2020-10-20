
paths = {
            'h5':'Dataset/h5Dataset/h5set.h5py', 
            'dataset':'Dataset/images',
            'model':'Models/saved_models',
            'current_model':'Models/saved_models/YEET8.h5',
            'test_img_path':'Dataset/FullJCNN2013',
            'large': 'Models/saved_models/large200.h5',
            'medium':'Models/saved_models/medium128.h5',
            'small':'Models/saved_models/small32.h5',
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

def get_dataset_path()->str:
    """Returns the path to the dataset

    Returns:
        str: the path to the dataset 
    """
    return paths.get("dataset")

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

def get_test_model_paths()->tuple:
    return paths.get("large"), paths.get("medium"), paths.get("small")