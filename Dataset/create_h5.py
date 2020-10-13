import h5py
import numpy as np
import os
import io
from PIL import Image
import dask.array as da

def CreateH5(h5Path, dataset_path):
    h5 = h5py.File(h5Path, 'w')

    for i in os.listdir(dataset_path):
        group_name = os.path.join(dataset_path, i)

        group = h5.create_group(group_name)
        print("Group: ", i)

        for j in os.listdir(group_name):
            img_path = os.path.join(group_name, j)

            if img_path.endswith('.ppm'):
                with open(img_path, 'rb') as image:
                    data = image.read()
    
                data_np = np.asarray(data)
                data_set = group.create_dataset(j, data=data_np)
    h5.close()