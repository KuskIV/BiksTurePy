import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from phase_one.find_ideal_model import train_and_eval_models_for_size
from global_paths import get_paths, get_training_set_noise_path, get_test_set_noise_path, get_h5_train
from Noise_Generators.noise_main import Filter,premade_single_filter,apply_multiple_filters
from Dataset.load_h5 import h5_object
from phase_one.find_ideal_model import  get_best_phase_one_model
from general_image_func import auto_reshape_images,changeImageSize,rgba_to_rgb,convert_between_pill_numpy

def save_dataset(dataset, path):
    imgs = [imgs[0] for imgs in dataset]
    lables = [label[1] for label in dataset]
    filters = [noise_filter[2] for noise_filter in dataset]

    create_dir(path)
    for label in lables:
        create_dir(f'{path}/{label}')
        for i in range(len(imgs)):
            pass



def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def train_noise_model(h5_obj,model_object,data_split,filters):
    dataset = generate_noise_dataset(h5_obj,data_split,filters,model_object.img_shape)
    save_dataset(dataset[0], get_training_set_noise_path())
    save_dataset(dataset[1], get_test_set_noise_path())
    train_model(dataset,model_object) #trains model on new dataset

def generate_noise_dataset(h5_obj, dataset_split, filters,image_size, lazy_split=10, lazy_start=0):
    original_images, original_labels, test_images, test_labels = h5_obj.shuffle_and_lazyload(lazy_start, lazy_split)#fetch and shuffle the data
    training_set = add_noise((original_images,original_labels), filters, image_size)#adds the noise to the images in linear
    test_set = add_noise((test_images,test_labels), filters, image_size)#adds the noise to the images in linear
    return training_set,test_set

def load_filters():
    F = premade_single_filter('fog')
    R = premade_single_filter('rain')
    S = premade_single_filter('snow')
    D = premade_single_filter('day')
    N = premade_single_filter('night')
    #dict = [{'fog':F}, {'night':N},{'rain':R},{'snow':S},{'day':D},{'night':N}]
    dict = {'fog':F,'rain':R,'snow':S,'day':D,'night':N}
    return dict

def add_noise(imgs, noise_filter, image_size):
    pil_imgs = convert_between_pill_numpy(imgs[0] * 255,mode='numpy->pil')
    lables = [lable for lable in imgs[1]]
    image_tuples = apply_multiple_filters((pil_imgs,lables),filters = noise_filter, mode='linear', KeepOriginal=True)
    RGB_img = [changeImageSize(image_size[0],image_size[1],im[0].convert('RGB')) for im in image_tuples]
    #numpy_imgs = convert_between_pill_numpy(RGB_img,mode='pil->numpy')
    for i in range(len(RGB_img)):
        image_tuples[i] = (image_tuples[i][0],image_tuples[i][2],image_tuples[i][1])
    return image_tuples

def train_model(data_set, model_object, epochs = 10, save_model = True):
    
    train_set,test_set = data_set #*list(tuple(image,class,filter,predicted_class)) assumed constrution of train_set/test_set
    train_imgs = [tuplen[0] for tuplen in train_set]
    train_lables = [tuplen[1] for tuplen in train_set]

    test_imgs = [tuplen[0] for tuplen in test_set]
    test_lables = [tuplen[1] for tuplen in test_set]

    train_and_eval_models_for_size(model_object.img_shape ,model_object.model,train_imgs,train_lables,test_imgs,test_lables,epochs=epochs)

def test_model(model,test_path):
    pass
def create_csv():
    pass

def qucik_debug():#TODO insert params
    h5_path = get_h5_train()
    training_split = 1
    h5_obj = h5_object(h5_path, training_split=training_split)
    ideal_model = get_best_phase_one_model(h5_obj.class_in_h5)
    train_noise_model(h5_obj, ideal_model,0.6,load_filters())
qucik_debug()