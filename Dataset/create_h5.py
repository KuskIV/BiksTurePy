import h5py
import numpy as np
import os
import io
from PIL import Image
import os.path
from os import path
from tqdm import tqdm
from tqdm import trange

def generate_h5(h5Path, dataset_path):

    print(f"\n\n\nA new H5PY file is about to be created.\n    Dataset path: {dataset_path}\n    H5PY Path: {h5Path}\n\n")
    if path.exists(h5Path):
        print("The H5PY file has been located, and is being overwritten")
        h5 = h5py.File(h5Path, 'w')
    else:
        h5 = h5py.File(h5Path, 'a')
        print("The H5PY file could not be found at the path, and a new is created")
    
    done = len(os.listdir(dataset_path))
    classes = trange(len(os.listdir(dataset_path)), desc='Class stuff', leave=True)
    for i in classes:
        classes.set_description(f"Class {i + 1} / {done}")
        classes.refresh()
        
        group_name = os.path.join(dataset_path, os.listdir(dataset_path)[i])

        group = h5.create_group(group_name)

        for j in os.listdir(group_name):
            img_path = os.path.join(group_name, j)
            if img_path.endswith('.ppm'):
                img = Image.open(img_path)
                data = np.asarray(img)

                data_set = group.create_dataset(j, data=data)
    h5.close()