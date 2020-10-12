import h5py
import numpy as np
import os
import io
from PIL import Image

h5Path = "h5Dataset/dataset.hdf5"
dataset_path = "Images/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/"





def CreateH5():
    h5 = h5py.File(h5Path, 'a')

    for i in os.listdir(dataset_path):
        group_name = os.path.join(dataset_path, i)
        group = h5.create_group(group_name)
        print("Group: ", i)

        for j in os.listdir(group_name):
            img_path = os.path.join(group_name, j)

            with open(img_path, 'rb') as image:
                data = image.read()
                
            data_np = np.asarray(data)

            data_set = group.create_dataset(j, data=data_np)
    h5.close()

data = []
group = []

def SortGroups(name, obj):
    if isinstance(obj, h5py.Dataset):
        data.append(name)
    elif isinstance(obj, h5py.Group):
        group.append(name)

def ReadH5():
    h5 = h5py.File(h5Path, 'r')

    h5.visititems(SortGroups)

    for j in data:
        kk = np.array(h5[j])
        img = Image.open(io.BytesIO(kk))
        print('image size: ', img.size)

CreateH5()
    