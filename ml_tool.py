import tensorflow as tf
import math

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


def analyze_data(numpy_images, labels, images_per_class):
    min_img_size, avg_img_size, max_img_size = ananalyze_img_sizes(numpy_images)

    print("img size analysis: min, avg, max:")
    print(min_img_size)
    print(avg_img_size)
    print(max_img_size)



#def analyze_performance(test_images, test_labels, images_per_class):
