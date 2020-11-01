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
        
rename_folders(test_path)
rename_folders(train_path)

# for i in range(len(a)):
#     print(f"{a[i]} - {b[i]}")
    
# print(f"\n\n{len(a)} - {len(b)}")