import tensorflow as tf
import math
from statistics import quantiles
import re
import matplotlib.pyplot as plt
import seaborn as sbs

def makePrediction(model, image):
    img_reshaped = tf.reshape(image, (1, 32, 32, 3))
    return model.predict_step(img_reshaped)

def get_img_aspect_ratio(numpy_images):
    return [img.shape[0]/img.shape[1] for img in numpy_images]

def get_img_product_sizes(numpy_images):
    return [math.prod(img.shape[0:2]) for img in numpy_images]

def plot_kde(data_lists, label_lists):
    sbs.set_theme()
    for i in range(data_lists):
        sbs.kdeplot(data=data_lists[i], label_lists[i])

    plt.show()

def img_sizes_list_from_tuple_file(filelocation):
    number_re = re.compile('\d')
    image_sizes_as_tuples = []

    with fp = open(filelocation, 'r'):
        raw_image_sizes = fp.readlines()

    for raw_image_size in raw_image_sizes:
        size = number_re(raw_image_size)
        image_sizes_as_tuples.append(int(size[0]), int(size[1]))

    return image_sizes_as_tuples

def list_from_file(filelocation):

    with fp = open(filelocation, 'r'):
        data = fp.readlines()

    return [int(item) for item in data]

def ananalyze_img_sizes(numpy_images):
    img_sizes = [math.prod(img.shape[0:2]) for img in numpy_images]
    min_index = img_sizes.index(min(img_sizes))
    max_index = img_sizes.index(max(img_sizes))

    # average
    avg_width = 0
    avg_height = 0

    for image in numpy_images:
        avg_width += image.shape[0]
        avg_height += image.shape[1]

    avg_width = math.floor(avg_width/len(numpy_images))
    avg_height = math.floor(avg_height/len(numpy_images))

    avg_img_size = (avg_width, avg_height)

    return numpy_images[min_index].shape[0:2], avg_img_size, numpy_images[max_index].shape[0:2],

def analyze_img_res_prod_quantiles(numpy_images, num_of_qtiles = 10):
    img_sizes_prod = [math.prod(img.shape[0:2]) for img in numpy_images]
    qtiles = [q for q in quantiles(img_sizes_prod, n= num_of_qtiles)]
    return qtiles

def analyze_data(numpy_images, labels, images_per_class):
    min_img_size, avg_img_size, max_img_size = ananalyze_img_sizes(numpy_images)

    # print(f'minimum image size: {min_img_size}')
    # print(f'average image size: {avg_img_size}')
    # print(f'maximum image size: {max_img_size}')
    # num_of_qtiles = 10
    # product_quantiles = analyze_img_res_prod_quantiles(numpy_images, num_of_qtiles = num_of_qtiles)
    # print(f'split into {num_of_qtiles} quantiles')
    # print(product_quantiles)
    # rooted_quantiles = [(int(math.sqrt(size)), int(math.sqrt(size))) for size in product_quantiles]
    # print(f'rooted quantiales of {num_of_qtiles}')
    # print(rooted_quantiles)

    imgs_size_prod = get_img_product_sizes(numpy_images)
    imgs_aspect_ratio = get_img_aspect_ratio(numpy_images)

    data_lists = [imgs_size_prod, imgs_aspect_ratio]
    label_lists = ["product of image sizes", "image aspect ratio (W/H)"]

    for img in numpy_images:
        print(tuple(img.shape[0:2]))

    #plot_kde(data_lists, label_lists)







#def analyze_performance(test_images, test_labels, images_per_class):
