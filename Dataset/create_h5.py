import h5py
import numpy as np
import os
import io
from PIL import Image
import os.path
from os import path

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from progressbar import print_progressbar

def generate_h5(h5Path, dataset_path):

    print(f"A new H5PY file is about to be created.\n    Dataset path: {dataset_path}\nH5PY Path: {h5Path}")
    if path.exists(h5Path):
        print("The H5PY file has been located, and is being overwritten")
        h5 = h5py.File(h5Path, 'w')
    else:
        h5 = h5py.File(h5Path, 'a')
        print("The H5PY file could not be found at the path, and a new is created")

    print_progressbar(0, 43, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i in os.listdir(dataset_path):
        group_name = os.path.join(dataset_path, i)

        group = h5.create_group(group_name)
        print("Group: ", i)

        for j in os.listdir(group_name):
            img_path = os.path.join(group_name, j)
            printProgressBar(j + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
            if img_path.endswith('.ppm'):
                img = Image.open(img_path)
                data = np.asarray(img)

                data_set = group.create_dataset(j, data=data)
    h5.close()