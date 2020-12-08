import os,sys,inspect
import shutil
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from phase_one.find_ideal_model import train_and_eval_models_for_size
from global_paths import get_paths, get_training_set_noise_path, get_test_set_noise_path, get_h5_train, get_h5_test
from Noise_Generators.noise_main import Filter,premade_single_filter,apply_multiple_filters
from Dataset.load_h5 import h5_object
from Dataset.create_h5 import generate_h5
from phase_one.find_ideal_model import get_satina_gains_model_object_list
from general_image_func import auto_reshape_images,changeImageSize,rgba_to_rgb,convert_between_pill_numpy
from rename_folders import rename_and_add_folders

def save_dataset(dataset, path):
    #The tuple is exstected to be of the form (img,lable,filter) in a list
    imgs = [imgs[0] for imgs in dataset]
    lables = [label[1] for label in dataset]
    filters = [noise_filter[2] for noise_filter in dataset]
    # for i in range(len(imgs)):
    #     imgs[i].show()
    #     print(lables[i])
    create_dir(path) #creates a folder if non-exsistes
    for label in lables: #iterates over the class of the imgs
        create_dir(f'{path}/{label}')# then it creates a folder foreach of the classes
        
    for i in range(len(imgs)):
        imgs[i].save(f"{path}/{lables[i]}/{i}_{filters[i]}.ppm")#*saves the img as ppm, using the index and the filter used on it, sperates by and underscore for the ease of passing later.


def create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.mkdir(path)

def generate_dataset(h5_obj, dataset_split, filters,image_size, chungus, keep_original, lazy_end=1, lazy_start=0):
    original_images, original_labels, _, _ = h5_obj.shuffle_and_lazyload(lazy_start, lazy_end)#TODO this should be training and validation set instead. Since the test set should now be loaded in sperately
    training_set = add_noise((original_images,original_labels), filters, image_size, chungus, keep_original)#adds the noise to the images in linear
    return training_set #,test_set

def generate_and_save_dataset(h5_obj, data_split, filters, img_shape, data_set_noise_path, data_set_path, h5_noise_path, chungus, keep_original):
    dataset = generate_dataset(h5_obj,data_split,filters,img_shape, chungus, keep_original)#generates the dataset with noises on the training and validation set
    save_dataset(dataset, data_set_noise_path) #saves the dataset in the provided path
    rename_and_add_folders(data_set_noise_path, data_set_path)
    generate_h5(h5_noise_path,data_set_noise_path)

def generate_train_set(data_split, filters, img_shape, keep_original, train_noise_key='train_set_noise', h5_noise_key='h5_train_noise', train_key='train_set', chungus=4):
    h5_path = get_h5_train()
    h5_obj = h5_object(h5_path, training_split=1)
    generate_and_save_dataset(h5_obj, 
                                    data_split, 
                                    filters, 
                                    img_shape, 
                                    get_paths(train_noise_key), 
                                    get_paths(train_key), 
                                    get_paths(h5_noise_key),
                                    chungus,
                                    keep_original
                                    )

def generate_test_set(data_split, filters, img_shape, keep_original, test_noise_key='test_set_noise', h5_noise_key='h5_test_noise', test_key='test_set',chungus=4):
    h5_path = get_h5_test()
    h5_obj = h5_object(h5_path, training_split=1)
    generate_and_save_dataset(h5_obj, 
                                data_split, 
                                filters, 
                                img_shape, 
                                get_paths(test_noise_key), 
                                get_paths(test_key), 
                                get_paths(h5_noise_key),
                                chungus,
                                keep_original
                                )


def load_homo_filters():
    H = premade_single_filter('std_homo')
    return {'std_homo':H}

def load_fog_filters():
    F = premade_single_filter('fog')
    dict = {'fog':F}
    return dict

def load_dehaze_filters():
    D = premade_single_filter('de_fog15')
    dict = {'dehaze':D}
    return dict

def load_filters():
    F = premade_single_filter('foghomo')
    R = premade_single_filter('rainhomo')
    S = premade_single_filter('snowhomo')
    D = premade_single_filter('dayhomo')
    N = premade_single_filter('nighthomo')
    dict = {'fog':F,'night':N,'rain':R,'snow':S,'day':D,'night':N}
    # dict = {'fog':F}
    return dict


def add_noise(imgs, noise_filter, image_size, chungus, keep_original):
    pil_imgs = convert_between_pill_numpy(imgs[0] * 255,mode='numpy->pil') #converts numpy_img list to pill imges in a list
    lables = [lable for lable in imgs[1]] #extracts the lables for the images
    image_tuples = apply_multiple_filters((pil_imgs,lables),filters = noise_filter, mode='linear', chungus=chungus, KeepOriginal=keep_original) #applies the diffrent noises to the images in a linear distribution, based on the number of filters inputet
    RGB_img = [changeImageSize(image_size[0],image_size[1],im[0].convert('RGB')) for im in image_tuples]#TODO find out what this line is used for other that its length. potentialy a useless computation
    #numpy_imgs = convert_between_pill_numpy(RGB_img,mode='pil->numpy')
    for i in range(len(RGB_img)):
        image_tuples[i] = (image_tuples[i][0],image_tuples[i][2],image_tuples[i][1])#rearranges the img tuple and overwrites the old tuple
    return image_tuples

def train_model(data_set, model_object, epochs = 10, save_model = True):
    
    train_set,test_set = data_set #*list(tuple(image,class,filter,predicted_class)) assumed constrution of train_set and test_set

    #exstracts the imgs and labes from the tuple, these are extracted to be used in the train_and_eval.. method below
    train_imgs = [tuplen[0] for tuplen in train_set]
    train_lables = [tuplen[1] for tuplen in train_set]

    test_imgs = [tuplen[0] for tuplen in test_set]
    test_lables = [tuplen[1] for tuplen in test_set]

    train_and_eval_models_for_size(model_object.img_shape ,model_object.model,train_imgs,train_lables,test_imgs,test_lables,epochs=epochs)#!High potentiel to be patialy deprecated, and should propely be updated the a new form

def generate_noise_dataset(img_shape, keep_original, data_split=1):
    filters = load_fog_filters()
    generate_train_set(data_split, filters, img_shape, keep_original)
    generate_test_set(data_split, filters, img_shape, keep_original)

def generate_homo_dataset(img_shape, keep_original, data_split=1):
    filters = load_homo_filters()
    generate_train_set(data_split, filters,img_shape, keep_original, chungus=0, train_noise_key='train_set_homo', h5_noise_key='h5_train_homo')
    generate_test_set(data_split, filters, img_shape, keep_original, chungus=0, test_noise_key='test_set_homo', h5_noise_key='h5_test_homo')

def generate_noise_homo_dataset(img_shape, keep_original, data_split=1):
    filters = load_filters()
    generate_train_set(data_split, filters,img_shape, keep_original, chungus=0, train_noise_key='train_set_ideal_noise', h5_noise_key='h5_train_ideal_noise')
    generate_test_set(data_split, filters, img_shape, keep_original, chungus=0, test_noise_key='test_set_ideal_noise', h5_noise_key='h5_test_ideal_noise')

def generate_dehaze_dataset(img_shape, keep_original, data_split=1):
    filters = load_dehaze_filters()
    generate_train_set(data_split, filters,img_shape, keep_original, chungus=0, train_noise_key='train_set_dehaze', h5_noise_key='h5_train_dehaze')
    generate_test_set(data_split, filters, img_shape, keep_original, chungus=0, test_noise_key='test_set_dehaze', h5_noise_key='h5_test_dehaze')


def generate_datasets():
    img_shape = (200, 200)
    generate_homo_dataset(img_shape, False) 
    generate_noise_dataset(img_shape, True)
    generate_noise_homo_dataset(img_shape, False)
    generate_dehaze_dataset(img_shape, False)

if __name__ == "__main__":
    generate_datasets()
