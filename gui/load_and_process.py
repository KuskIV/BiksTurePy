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

def get_parameters(img_path):
    max_vals = 0
    min_vals = 0
    img_sizes = (0, 0)
    mean_vals = 0
    valid_output = False
    try:
        img_loaded = imageio.imread(img_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
    else:
        max_vals = img_loaded.max()
        min_vals = img_loaded.min()
        img_sizes = img_loaded.shape
        mean_vals = np.average(img_loaded)
        valid_output = True
    finally:
        return max_vals, min_vals, img_sizes, mean_vals, valid_output


def analyze_images(folder_path, img_names):
    mean_vals = []
    max_vals = []
    min_vals = []
    img_sizes = []
    
    for img in img_names:
        img_path = f"{folder_path}/{img}"
        max_val, min_val, img_size, mean_val, valid_output = get_parameters(img_path)
        
        if valid_output:
            mean_vals.append(mean_val)
            max_vals.append(max_val)
            min_vals.append(min_val)
            img_sizes.append(img_size)

    return mean_vals, max_vals, min_vals, img_sizes

def load_images_and_labels(folder_path):
    img_extension = '.ppm'
    img_names = os.listdir(folder_path)
    a_vals, b_vals = parse_and_verify_img_names(img_names, img_extension)
    mean_vals, max_vals, min_vals, img_sizes = analyze_images(folder_path, img_names)
    
    return a_vals, b_vals, mean_vals, max_vals, min_vals, img_sizes

def get_data_from_image(img_path):
    max_vals, min_vals, img_sizes, mean_vals, valid_output = get_parameters(img_path)
    try:
        h = img_sizes[0]
        w = img_sizes[1]
    except Exception as e:
        print(f"ERROR: {e}, {img_sizes}")
        raise Exception
    return [mean_vals, max_vals, min_vals, h, w], valid_output

def get_data(folder_path):
    return load_images_and_labels(folder_path)

if __name__ == "__main__":
    roni_bot_path = "Dataset/roni_bot"
    a_vals, b_vals, mean_vals, max_vals, min_vals, img_sizes = load_images_and_labels(roni_bot_path)
    
    for i in range(len(a_vals)):
        try:
            print(f"a = {a_vals[i]}, b = {b_vals[i]}, min: {min_vals[i]}, max: {max_vals[i]}, avg: {round(mean_vals[i], 2)}, size: {img_sizes[i]}")
        except IndexError as e:
            print(f"ERROR: {e}")
            raise IndexError