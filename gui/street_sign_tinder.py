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

def generate_label_set_numpy(a_vals, b_vals, l_count):
    label_set = np.empty((len(a_vals), l_count), dtype=float, order='C')
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

def generate_data_set_numpy(parameters, f_count):
    data_set = np.empty((len(parameters), f_count), dtype=float, order='C')
    
    for i in range(len(parameters)):
        try:
            for j in range(len(parameters[0])):
                data_set[i][j] = parameters[i][j]
        except Exception as e:
            print(f"ERROR: {e}")
            raise Exception
    return data_set

def show_image_with_parameters(parameters, img_path, save_path):
    config = {'a':parameters[0],'b':parameters[1],'cutoff':3}
    homo = homomorphic(config)
    img = homo.homofy(Image.open(img_path))
    img = img.resize((200, 200))
    img.save(save_path)

def new_name(yhat, id, img_extension):
    id = id.split('_')[0]
    return f"{id}_a_{round(float(yhat[0]),3)}_b_{round(float(yhat[1]),3)}{img_extension}"

if __name__ == "__main__":
    roni_bot_path = "Dataset/roni_bot"
    roni_output = "Dataset/roni_output"
    test_img_name = "1_a_1.0_b_0.0.ppm"
    dark_img = "0_a_1.209_b_0.209.ppm"
    
    input_path = f"{roni_bot_path}/{test_img_name}"
    output_path = f"{roni_output}/{test_img_name}"
    
    f_count = 4
    l_count = 2
    
    parameters, a_vals, b_vals = get_data(roni_bot_path)
    
    lable_set = generate_label_set_numpy(a_vals, b_vals, l_count)
    data_set = generate_data_set_numpy(parameters, len(parameters[0]))
    
    n_samples = len(a_vals)

    n_inputs, n_outputs = len(data_set[0]), len(lable_set[0])
    # get model
    model = get_model(n_inputs, n_outputs)
    # fit the model on all data
    model.fit(data_set, lable_set, verbose=0, epochs=100)
    
    
    for img in os.listdir(roni_bot_path):
        input_path = f"{roni_bot_path}/{img}"
        
        
        test_parameters = get_data_from_image(input_path)
        
        newX = asarray([test_parameters])
        yhat = model.predict(newX)
        print('Predicted: %s' % yhat[0])
        
        output_name = new_name(yhat[0], img, '.ppm')
        output_path = f"{roni_output}/{output_name}"
        
        try:
            show_image_with_parameters(yhat[0], input_path, output_path)
        except Exception as e:
            print(f"ERROR: {e}")
    
    
    print(test_parameters)
    # importance = model.importance_mean
    # # summarize feature importance
    # for i,v in enumerate(importance):
    #     print('Feature: %0d, Score: %.5f' % (i,v))
    # # plot feature importance
    # pyplot.bar([x for x in range(len(importance))], importance)
    # pyplot.show()
    
    