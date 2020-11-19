# use mlp for prediction on multi-output regression
from numpy import asarray
from sklearn.datasets import make_regression
from keras.models import Sequential
from keras.layers import Dense
from load_and_process import get_data, get_data_from_image
import numpy as np
from PIL import Image

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from Noise_Generators.homomorphic_filtering import homomorphic

# get the dataset
def get_dataset(n_samples, n_features):
	data, lables = make_regression(n_samples=n_samples, n_features=n_features, n_informative=5, n_targets=2, random_state=2)
	return data, lables

# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
	model.compile(loss='mae', optimizer='adam')
	return model

def generate_label_set(a_vals, b_vals):
    data_set = np.zeros(len(a_vals))
    label_set = []
    for i in range(len(a_vals)):
        try:
            # data_to_append = np.array([a_vals[i], b_vals[i]]) 
            # data_set[i] = data_to_append
            label_set.append([a_vals[i], b_vals[i]])
        except Exception as e:
            print(f"ERROR: {e}")
            raise Exception
    return [label_set]

def generate_label_set_numpy(a_vals, b_vals):
    label_set = np.empty((len(a_vals), 2), dtype=float, order='C')
    for i in range(len(a_vals)):
        try:
            # data_to_append = np.array([a_vals[i], b_vals[i]]) 
            # data_set[i] = data_to_append
            label_set[i][0] = a_vals[i]
            label_set[i][1] = b_vals[i]
            # np.append(label_set[i], [a_vals[i], b_vals[i]], axis=None)
            
        except Exception as e:
            print(f"ERROR: {e}")
            raise Exception
    return label_set

def generate_data_set(mean_vals, max_vals, min_vals, img_sizes):
    data_set = []
    for i in range(len(mean_vals)):
        try:
            data_set.append([mean_vals[i], max_vals[i], min_vals[i], img_sizes[i][0], img_sizes[i][1], img_sizes[i][2]])
        except Exception as e:
            print(f"ERROR: {e}")
            raise Exception
    return [data_set]

def generate_data_set_numpy(mean_vals, max_vals, min_vals, img_sizes):
    data_set = np.empty((len(mean_vals), 5), dtype=float, order='C')
    for i in range(len(mean_vals)):
        try:
            data_set[i][0] = mean_vals[i]
            data_set[i][1] = max_vals[i]
            data_set[i][2] = min_vals[i]
            data_set[i][3] = img_sizes[i][0]
            data_set[i][4] = img_sizes[i][1]
            # data_set.append([mean_vals[i], max_vals[i], min_vals[i], img_sizes[i][0], img_sizes[i][1], img_sizes[i][2]])
        except Exception as e:
            print(f"ERROR: {e}")
            raise Exception
    return data_set

def show_image_with_parameters(parameters, img_path):
    config = {'a':parameters[0],'b':parameters[1],'cutoff':3}
    homo = homomorphic(config)
    img = homo.homofy(Image.open(img_path))
    img = img.resize((200, 200))
    img.show()



if __name__ == "__main__":
    roni_bot_path = "Dataset/roni_bot"
    a_vals, b_vals, mean_vals, max_vals, min_vals, img_sizes = get_data(roni_bot_path)
    
    lable_set = generate_label_set_numpy(a_vals, b_vals)
    data_set = generate_data_set_numpy(mean_vals, max_vals, min_vals, img_sizes)
    
    n_samples = len(a_vals)

    n_inputs, n_outputs = len(data_set[0]), len(lable_set[0])
    # get model
    model = get_model(n_inputs, n_outputs)
    # fit the model on all data
    model.fit(data_set, lable_set, verbose=0, epochs=100)
    
    test_img_path = "Dataset/roni_bot/1_a_1.0_b_0.0.ppm"
    test_parameters, valid_parameter = get_data_from_image(test_img_path)
    
    if not valid_parameter:
        raise Exception('test_img_path not valid')
    
    newX = asarray([test_parameters])
    yhat = model.predict(newX)
    print('Predicted: %s' % yhat[0])
    
    show_image_with_parameters(yhat[0], test_img_path)
    print(test_parameters)
    importance = model.coef_[0]
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()
    
    