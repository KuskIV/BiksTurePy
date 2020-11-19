import numpy as np
import os
from PIL import Image
import imageio

def verify_img_names(img_names, img_extension):
    for img in img_names:
        if not img.endswith(img_extension):
            raise TypeError(f"The image {img} is not in the correct syntax. should end with '{img_extension}'.")
        if len(img.split('_')) != 5:
            raise TypeError(f"The image {img} is not in the correct syntax. Should contain four '_'.")

def parse_and_verify_img_names(img_names, img_extension):
    verify_img_names(img_names, img_extension)
    a_vals = [float(x.split('_')[2]) for x in img_names]
    b_vals = [float(x.split('_')[4].replace(img_extension, '')) for x in img_names]
    return a_vals, b_vals

def analyze_images(folder_path, img_names):
    mean_vals = []
    max_vals = []
    min_vals = []
    img_sizes = []
    
    for img in img_names:
        try:
            img_path = f"{folder_path}/{img}"
            img_loaded = imageio.imread(img_path)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
        else:
            max_vals.append(img_loaded.max())
            min_vals.append(img_loaded.min())
            img_sizes.append(img_loaded.shape)
            mean_vals.append(np.average(img_loaded))

    return mean_vals, max_vals, min_vals, img_sizes

def load_images_and_labels(folder_path):
    img_extension = '.ppm'
    img_names = os.listdir(folder_path)
    a_vals, b_vals = parse_and_verify_img_names(img_names, img_extension)
    mean_vals, max_vals, min_vals, img_sizes = analyze_images(folder_path, img_names)
    
    return a_vals, b_vals, mean_vals, max_vals, min_vals, img_sizes

if __name__ == "__main__":
    roni_bot_path = "Dataset/roni_bot"
    a_vals, b_vals, mean_vals, max_vals, min_vals, img_sizes = load_images_and_labels(roni_bot_path)
    
    for i in range(len(a_vals)):
        try:
            print(f"a = {a_vals[i]}, b = {b_vals[i]}, min: {min_vals[i]}, max: {max_vals[i]}, avg: {round(mean_vals[i], 2)}, size: {img_sizes[i]}")
        except IndexError as e:
            print(f"ERROR: {e}")
            raise IndexError