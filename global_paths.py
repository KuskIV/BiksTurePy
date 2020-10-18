
paths = {
            'h5':'Dataset/h5Dataset/h5set.h5py', 
            'dataset':'Dataset/images',
            'model':'Models/saved_models',
            'current_model':'Models/saved_models/YEET8.h5',
            'test_img_path':'Dataset/FullJCNN2013'
    }

def get_paths(key:str)->str:
    if key in paths:
        return paths.get(key)
    else:
        print(f"ERROR: The key for path {key} is not defined.")
        return -1

def get_h5_path()->str:
    return paths.get("h5")

def get_dataset_path()->str:
    return paths.get("dataset")

def get_current_model_path()->str:
    return paths.get("current_model")

def get_test_img_path()->str:
    return paths.get("test_img_path")

def get_model_path()->str:
    return paths.get("model") 