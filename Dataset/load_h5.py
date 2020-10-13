import h5py
import numpy as np
import os
import io
from PIL import Image
import dask.array as da
from os import path
import os.path
import sys

data = []
group = []

def sort_groups(name, obj):
    if isinstance(obj, h5py.Dataset):
        data.append(name)
    elif isinstance(obj, h5py.Group):
        group.append(name)


def read_h5(h5_path): # Not yet implemented correctly
    h5 = h5py.File(h5_path, 'r')
    print(h5['Dataset']['images'])
    #print(test)
            
    h5.visititems(sort_groups)

    print(f"groups: {len(group)}")
    print(f"data: {len(data)}")

    #for j in data:
    #    kk = np.array(h5[j])
    #    img = Image.open(io.BytesIO(kk))
    #    print('image size: ', img.size)

def get_h5(h5_path):
    if not path.exists(h5_path):
        print(f"The path for the h5 file does not exist ({h5_path}). The program has exited.")
        sys.exit()
    else:
        h5 = h5py.File(h5_path, 'r')
        h5.visititems(sort_groups)
        return h5

def lazyload_h5(h5, current_split, max_split):
    isLast = current_split == max_split - 1

    print(f"groups: {len(group)}")
    print(f"data: {len(data)}")

    

    return [], [], [], []