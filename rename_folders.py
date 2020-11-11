from global_paths import get_training_set_path, get_test_set_path

import os

test_path = get_test_set_path()
train_path = get_training_set_path()

print(test_path)
print(train_path)

def rename_folders(path):
    folder_names = []

    for filename in os.listdir(path):
        folder_names.append(filename)
    
    folder_names.sort()
    
    # return folder_names
        
    for i in range(len(folder_names)):
        # print(folder_names[i])
        os.rename(f"{path}/{folder_names[i]}", f"{path}/{i}")

def add_folders(train_path, test_path):
    test_folder_names = []
    train_folder_names = []
    
    for filename in os.listdir(train_path):
        train_folder_names.append(filename)
    
    for filename in os.listdir(test_path):
        test_folder_names.append(filename)
    
    for train_folder in train_folder_names:
        if not train_folder in test_folder_names:
            os.mkdir(f"{test_path}/{train_folder}")
            test_folder_names.append(train_folder)
    
    for test_folder in test_folder_names:
        if not test_folder in train_folder_names:
            os.mkdir(f"{train_path}/{test_folder}")
            train_folder_names.append(test_folder)
            
def rename_and_add_folders(train_path,test_path):
    add_folders(train_path, test_path)

    # rename_folders(train_path)
    # rename_folders(test_path)
    
if __name__ == "__main__":
    add_folders(train_path, test_path)

    rename_folders(train_path)
    rename_folders(test_path)