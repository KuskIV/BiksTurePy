import h5py
import numpy as np
import os
import io
from PIL import Image
import dask.array as da

h5Path = "h5Dataset/dataset.hdf5"
dataset_path = "images/"

def CreateH5():
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

data = []
group = []

def SortGroups(name, obj):
    if isinstance(obj, h5py.Dataset):
        data.append(name)
    elif isinstance(obj, h5py.Group):
        group.append(name)

def LazyRead():
    big_chungus = (1000, 100)
    f = h5py.File(h5Path, 'r')
    data = f[dataset_path]
    print(data.shape)

    #x = da.from_array(h5Path, chunks=big_chungus)



def ReadH5():
    h5 = h5py.File(h5Path, 'r')

    test = h5.get('images').get('00000')
    test = h5['images']['00000']['00000_00000.ppm']
    print(test)
            
    h5.visititems(SortGroups)

    #print(f"groups: {len(group)}")
    #print(f"data: {len(data)}")

    for j in data:
        kk = np.array(h5[j])
        img = Image.open(io.BytesIO(kk))
        print('image size: ', img.size)

#CreateH5Updated()

#ReadH5()

#LazyRead()

CreateH5()
    