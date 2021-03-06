import re
import os

fileExstension = '.ppm'




def get_dataset_placements(dataset_path: str)->tuple:
    """Assumes that the exact FULLIJCNN2013 folder is in the root from
       http://benchmark.ini.rub.de/?section=gtsdb&subsection=dataset.
       Returns a tuple, where first item is a list with the placement of each
       image in the FULLIJCNN2013 dataset of the detected images
       and store in this format [image_path, label]. Second tuple item is number of images
       per classification/label"""

    dataset_placements = []
    images_per_class = []
    with os.scandir(dataset_path) as dir:
        for entry in dir:
            if entry.is_dir(): # directories with detected signs
                with os.scandir(entry.path) as detect_dir: # here all files are the .ppm images, the dir name indicates its label
                    num_of_images = 0
                    for ppm_image in detect_dir:
                        if ppm_image.path.endswith(fileExstension):
                            dataset_placements.append([ppm_image.path, int(entry.name)])
                            num_of_images += 1
                    images_per_class.append(num_of_images)

    return dataset_placements, images_per_class

