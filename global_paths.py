
paths = {
    'h5': 'Dataset/h5Dataset/h5set.h5py',
    'h5_train': 'Dataset/h5Dataset/h5_train.h5py',
    'h5_test': 'Dataset/h5Dataset/h5_test.h5py',
    'train_set': 'Dataset/satina_gains_images/satina_trains',
    'test_set': 'Dataset/satina_gains_images/satina_tests',
    'train_set_noise': 'Dataset/satina_gains_images/satina_trains_noise',
    'test_set_noise': 'Dataset/satina_gains_images/satina_tests_noise',
    'h5_train_noise': 'Dataset/h5Dataset/h5_train_noise.h5py',
    'h5_test_noise': 'Dataset/h5Dataset/h5_test_noise.h5py',
    'model': 'Models/saved_models',
    'ex_one_ideal': 'Models/saved_models/ex_one_ideal.h5',
    'large': 'Models/saved_models/large200.h5',
    'medium': 'Models/saved_models/medium128.h5',
    'small': 'Models/saved_models/small32.h5',
    'phase_one_csv': 'phase_one/csv_data',
    'phase_two_csv': 'phase_two/csv_output',
    'satina_avg':'Models/saved_models/satina_avg.h5',
    'satina_median':'Models/saved_models/satina_median.h5',
    'satina_mode':'Models/saved_models/satina_mode.h5',
    'satina_avg_norm':'Models/saved_models/satina_avg_norm.h5',
    'satina_median_norm':'Models/saved_models/satina_median_norm.h5',
    'satina_mode_norm':'Models/saved_models/satina_mode_norm.h5',
    'satina_avg_noise':'Models/saved_models/satina_avg_noise.h5',
    'satina_median_noise':'Models/saved_models/satina_median_noise.h5',
    'satina_mode_noise':'Models/saved_models/satina_mode_noise.h5',
    # 'category_csv':'Dataset/test.csv',
    'category_csv':'Dataset/ETSD_Adjusted/test.csv',
    # 'category_csv':'Dataset/class_descrip.csv',
    'txt_file': 'labels.txt'
}

def get_category_csv():
    return paths['category_csv']

def get_paths(key: str) -> str:
    """Given a key, the value for that key is returned

    Args:
        key (str): The input key

    Returns:
        str: The corresponding value
    """
    if key in paths:
        return paths.get(key)
    else:
        raise KeyError(f"The key {key} does not exist in global path dictionary")


def get_h5_path() -> str:
    """Returns the path to the h5py file

    Returns:
        str: the path to the h5py file
    """
    return paths.get("h5")


def get_h5_train() -> str:
    return paths.get('h5_train')


def get_h5_test() -> str:
    return paths.get('h5_test')

def get_h5_train_noise() -> str:
    return paths.get('h5_train_noise')


def get_h5_test_noise() -> str:
    return paths.get('h5_test_noise')


def get_training_set_path() -> str:
    return paths.get('train_set')


def get_training_set_noise_path() -> str:
    return paths.get('train_set_noise')


def get_data_set_path() -> str:
    """Returns the path to the dataset

    Returns:
        str: the path to the dataset
    """
    return get_training_set_path()


def get_test_set_path() -> str:
    return paths.get('test_set')


def get_test_set_noise_path() -> str:
    return paths.get('test_set_noise')


def get_current_model_path() -> str:
    """Returns the path to the current model

    Returns:
        str: the path to the current model
    """
    return paths.get("current_model")


def get_test_img_path() -> str:
    """Returns the path to the test images

    Returns:
        str: the path to the test images
    """
    return paths.get("test_img_path")


def get_model_path() -> str:
    """Returns the path to the model

    Returns:
        str: the path to the model
    """
    return paths.get("model")


def get_large_model_path() -> str:
    return paths.get("large")


def get_medium_model_path() -> str:
    return paths.get('medium')


def get_small_model_path() -> str:
    return paths.get('small')


def get_satina_model_mode_path() -> str:
    return paths.get('satina_mode')


def get_satina_model_avg_path() -> str:
    return paths.get('satina_avg')


def get_satina_model_median_path() -> str:
    return paths.get('satina_median')


def get_satina_model_mode_path_norm() -> str:
    return paths.get('satina_mode_norm')


def get_satina_model_avg_path_norm() -> str:
    return paths.get('satina_avg_norm')


def get_satina_model_median_path_norm() -> str:
    return paths.get('satina_median_norm')

def get_satina_model_mode_path_noise() -> str:
    return paths.get('satina_mode_noise')


def get_satina_model_avg_path_noise() -> str:
    return paths.get('satina_avg_noise')

def get_satina_model_median_path_noise() -> str:
    return paths.get('satina_median_noise')

def get_phase_two_csv(name:str=None, extension:str=None)->str:
    base_path=get_paths('phase_two_csv')
    return f'{base_path}/phase2_{name}.csv' if extension==None else f'{base_path}/{extension}/phase2_{name}.csv'