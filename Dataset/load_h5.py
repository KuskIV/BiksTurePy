import h5py
import numpy as np
import os
import io
from PIL import Image
import dask.array as da

data = []
group = []

def sort_groups(name, obj):
    if isinstance(obj, h5py.Dataset):
        data.append(name)
    elif isinstance(obj, h5py.Group):
        group.append(name)


def read_h5(h5_path): # Not yet implemented correctly
    h5 = h5py.File(h5_path, 'r')

    test = h5.get('images').get('00000')
    test = h5['images']['00000']['00000_00000.ppm']
    print(test)
            
    h5.visititems(sort_groups)

    print(f"groups: {len(group)}")
    print(f"data: {len(data)}")

    #for j in data:
    #    kk = np.array(h5[j])
    #    img = Image.open(io.BytesIO(kk))
    #    print('image size: ', img.size)