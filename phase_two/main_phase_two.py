import Noise_Generators.noise_main as noise
from Noise.generators.noise_main import Filter,premade_single_filter,apply_multiple_filters
from Dataset.load_h5 import h5_object
from global_paths import get_test_model_paths, get_paths, get_h5_test, get_h5_train
from Models.test_model import partial_accumilate_distribution, print_accumilate_distribution, make_prediction


def phase_2_1(model, h5path, lazysplit, image_size):
    h5 = h5_object(h5path, training_split=dataset_split)
    test = []
    for j in range(lazy_split):
        original_images, original_labels, _, _ = h5_obj.shuffle_and_lazyload(j, lazy_split) #TODO need variant of this that does not generate test set or shuffle
        image_tuples = add_noise((original_images,original_labels)) #tuple(image,class,filter)
        for i in range(len(image_tuples)):
            prediction = make_prediction(model, image_tuples[i][0], (image_size[0], image_size[1], 3))
            predicted_label = np.argmax(prediction) #Get the class with highest liklyhood of the predictions
            image_tuples[i] = image_tuples[i]+(predicted_label) #concatanate two tuples to create new tuple , which replacess the old one
        test.extend(image_tuples)
    convert_to_csv(test)

def load_filters():
    F = premade_single_filter('fog')
    R = premade_single_filter('rain')
    S = premade_single_filter('snow')
    D = premade_single_filter('day')
    N = premade_single_filter('night')
    dict = {'fog':F,'rain':R,'snow':S,'day':D,'night':N}
    return dict

def add_noise(imgs):
    return apply_multiple_filters(imgs,filters = load_filters(), mode='rand', KeepOriginal=True)

def QuickDebug():

    large_model_path, medium_model_path, small_model_path,belgium_model_path = get_test_model_paths()
        
    #large_model = tf.keras.models.load_model(large_model_path)
    #medium_model = tf.keras.models.load_model(medium_model_path)
    # small_model = tf.keras.models.load_model(small_model_path)
    belgium_model = tf.keras.models.load_model(belgium_model_path)
    phase_2_1(belgium_model,,10,(32,32))

def convert_to_csv(values):
    pass