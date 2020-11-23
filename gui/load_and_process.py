import numpy as np
import os
from PIL import Image
import imageio
from skimage import io
import cv2
import matplotlib.pyplot as plt

def pls_work(img_path):
    imgog = Image.open(img_path)
    imgYCC = imgog.convert('YCbCr')
    ycc_list = list(imgYCC.getdata())
    ycc_list = np.reshape(imgYCC, (imgog.size[1], imgog.size[0], 3))
    ycc_list.astype(np.uint8)

    y = Image.fromarray(ycc_list[:,:,0], "L")
    return find_intensity(y)
    

def find_intensity(image): 
    w,h = image.size
    w_c = int(w/2)
    h_c = int(h/2)
    h_p = int(h * 0.1)
    w_p = int(w * 0.1)
    total = 0
    max_pix = 0
    smallest_pepe = 0
    kernal_size = h_p * w_p
    n = 0
    for i in range(0, w_p):
        for j in range(0, h_p):
            total += image.getpixel((i+w_c,j+h_c))
            if max_pix < image.getpixel((i+w_c,j+h_c)): 
                max_pix = image.getpixel((i+w_c,j+h_c))
            if smallest_pepe > image.getpixel((i+w_c,j+h_c)): 
                smallest_pepe = image.getpixel((i+w_c,j+h_c))
    mean_av = total / kernal_size
    return mean_av, max_pix

def cluster(img_path):
    img = io.imread(img_path)[:, :, :-1]
    average = img.mean(axis=0).mean(axis=0)
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.cv2.TERM_CRITERIA_EPS + cv2.cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    avg_patch = np.ones(shape=img.shape, dtype=np.uint8)*np.uint8(average)
    dominant = palette[np.argmax(counts)]
    
    return dominant
    
    # print(dominant, " DOMINANT")
    # print(palette, " PALETTE")
    # print(counts, " COUNTS")

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
    dominant = (0, 0, 0)
    mean_vals = 0
    mean = 0
    max_pp = 0
    try:
        img_loaded = imageio.imread(img_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
    else:
        dominant = cluster(img_path)
        mean, max_pp = pls_work(img_path)
        mean_vals = np.average(img_loaded)
    finally:
        return mean_vals, dominant[0], dominant[1], dominant[2], mean, max_pp


def analyze_images(folder_path, img_names):
    parameters = []
    
    for img in img_names:
        img_path = f"{folder_path}/{img}"
        parameters.append(get_parameters(img_path))

    return parameters

def load_images_and_labels(folder_path):
    img_extension = '.ppm'
    img_names = os.listdir(folder_path)
    a_vals, b_vals = parse_and_verify_img_names(img_names, img_extension)
    parameters = analyze_images(folder_path, img_names)
    
    return parameters, a_vals, b_vals

def get_data_from_image(img_path):
    return get_parameters(img_path)

def get_data(folder_path):
    return load_images_and_labels(folder_path)

if __name__ == "__main__":
    roni_bot_path = "Dataset/roni_bot"
    img_test_path = "Dataset/roni_bot/1_a_1.0_b_0.0.ppm"
    cluster(img_test_path)
    
    img_test_path = "Dataset/roni_bot/139_a_1.0_b_0.0.ppm"
    cluster(img_test_path)
    # a_vals, b_vals, mean_vals, max_vals, min_vals, img_sizes = load_images_and_labels(roni_bot_path)
    
    # for i in range(len(a_vals)):
    #     try:
    #         print(f"a = {a_vals[i]}, b = {b_vals[i]}, min: {min_vals[i]}, max: {max_vals[i]}, avg: {round(mean_vals[i], 2)}, size: {img_sizes[i]}")
    #     except IndexError as e:
    #         print(f"ERROR: {e}")
    #         raise IndexError