import Noise_Generators.noise_main as noise
from Noise.generators.noise_main import Filter,premade_single_filter,apply_multiple_filters
from Dataset.load_h5 import h5_object
from Models.test_model import partial_accumilate_distribution, print_accumilate_distribution, make_prediction


def phase_2_1(model, h5path, lazysplit, image_size):
    h5 = h5_object(h5path, training_split=dataset_split)
    for j in range(lazy_split):
        original_images, original_labels, _, _ = h5_obj.shuffle_and_lazyload(j, lazy_split) #TODO need variant of this that does not generate test set
        image_tuple= add_noise((original_images,original_labels))
        prediction = make_prediction(model, image_dataset[j].copy(), (image_size[0], image_size[1], 3))
        predicted_label = np.argmax(prediction)

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