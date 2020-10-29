import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from phase_one.find_ideal_model import train_and_eval_models_for_size
from global_paths import get_paths
from Noise_Generators.noise_main import Filter,premade_single_filter,apply_multiple_filters
from Dataset.load_h5 import h5_object
from general_image_func import auto_reshape_images,changeImageSize,rgba_to_rgb,convert_between_pill_numpy


def train_noise_model(h5_path,model,data_split,filters,image_size):
    dataset = generate_noise_dataset(h5_path,data_split,filters,image_size)
    train_model(dataset,model,image_size) #trains model on new dataset


def generate_noise_dataset(h5_path, dataset_split, filters,image_size, lazy_split=1, lazy_start=0):
    h5_obj = h5_object(h5_path, training_split=dataset_split)#instantiat h5 object images
    original_images, original_labels, test_images, test_labels = h5_obj.shuffle_and_lazyload(lazy_start, lazy_split)#fetch and shuffle the data
    training_set = add_noise((original_images,original_labels), filters, image_size)#adds the noise to the images in linear
    test_set = add_noise((test_images,test_labels), filters, image_size)#adds the noise to the images in linear
    return training_set,test_images

def load_filters():
    F = premade_single_filter('fog')
    R = premade_single_filter('rain')
    S = premade_single_filter('snow')
    D = premade_single_filter('day')
    N = premade_single_filter('night')
    dict = [{'fog':F}, {'night':N},{'rain':R},{'snow':S},{'day':D},{'night':N}]
    #dict = {'fog':F,'rain':R,'snow':S,'day':D,'night':N}
    return dict

def add_noise(imgs, noise_filter, image_size):
    pil_imgs = convert_between_pill_numpy(imgs * 255,mode='numpy->pil')
    image_tuples = apply_multiple_filters(imgs,filters = noise_filter, mode='linear', KeepOriginal=True)
    return convert_between_pill_numpy([changeImageSize(image_size[0],image_size[1],im[0].convert('RGB')) for im in image_tuples],mode='pil->numpy')

def train_model(data_set, model, size, epochs = 10, save_model = True):
    
    train_set,test_set = data_set #*list(tuple(image,class,filter,predicted_class)) assumed constrution of train_set/test_set
    train_imgs = [tuplen[0] for tuplen in train_set]
    train_lables = [tuplen[1] for tuplen in train_set]

    test_imgs = [tuplen[0] for tuplen in test_set]
    test_lables = [tuplen[1] for tuplen in test_set]

    train_and_eval_models_for_size(size,model,yeet,train_imgs,train_lables,test_imgs,test_lables,epochs=epochs)#TODO remove yeet when changes to function is done

def test_model(model,test_path):
    pass
def create_csv():
    pass

def qucik_debug():#TODO insert params
    #train_noise_model()