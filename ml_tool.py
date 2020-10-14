import tensorflow as tf
import math
from statistics import quantiles

def makePrediction(model, image):
    img_reshaped = tf.reshape(image, (1, 32, 32, 3))
    return model.predict_step(img_reshaped)

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

    print(f'minimum image size: {min_img_size}')
    print(f'average image size: {avg_img_size}')
    print(f'maximum image size: {max_img_size}')
    num_of_qtiles = 10
    product_quantiles = analyze_img_res_prod_quantiles(numpy_images, num_of_qtiles = num_of_qtiles)
    print(f'split into {num_of_qtiles} quantiles')
    print(product_quantiles)
    rooted_quantiles = [(int(math.sqrt(size)), int(math.sqrt(size))) for size in product_quantiles]
    print(f'rooted quantiales of {num_of_qtiles}')
    print(rooted_quantiles)



#def analyze_performance(test_images, test_labels, images_per_class):
