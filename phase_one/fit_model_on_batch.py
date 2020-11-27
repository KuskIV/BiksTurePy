import tensorflow as tf
from keras import backend as K
import numpy as np
from tensorflow.keras import datasets, layers, models
from copy import copy
import os,sys,inspect
import time
import math
from tqdm import tqdm
from tqdm import trange
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from Models.test_model import make_prediction

def zip_and_shuffle(img, lab):
    if len(img) != len(lab):
        raise IndexError("The image list and lable list does not have the same length")
    try:
        zip_test = [(img[i], lab[i]) for i in range(len(img))]
        np.random.shuffle(zip_test)
        img = [zip_test[i][0] for i in range(len(zip_test))] 
        lab = [zip_test[i][1] for i in range(len(zip_test))] 
    except Exception as e:
        print(f"ERROR: {e}")
        raise Exception
    return img, lab

def get_batch(images, lables, batch_size, noises, noise_method, shuffle=True,
            drop_last=True, augmentation=False):
    idx = len(images)
    
    
    if drop_last:
        n_batches = idx // batch_size
    else:
        n_batches = np.ceil(idx / batch_size).astype(np.int32)
    
    if shuffle:
        images, lables = zip_and_shuffle(images, lables)
    
    for b in range(n_batches):
        left_idx = b * batch_size
        right_idx = min((b+1)*batch_size, idx)
        img_batch, lab_batch = images[left_idx:right_idx], lables[left_idx:right_idx]

        if augmentation:
            try:
                img_batch = noise_method(img_batch, noises, batch_size)
            except Exception as e:
                print(f"ERROR: {e}")
                raise Exception

        yield img_batch, lab_batch



def apply_noise_evenly(img_batch, noises, batch_size):
    global_idx = 0
    aug_bs = batch_size // len(noises)
    
    for i, noise in enumerate(noises):
        for img in img_batch[i*aug_bs:i+1*aug_bs]:
            img_batch[global_idx] = noise + img
            global_idx += 1

    return img_batch

def should_early_stop(best_epoch, epoch, patience):
    return best_epoch + patience <= epoch

def sum_accuracy(right, wrong):
    if len(right) != len(wrong):
        raise IndexError("The list 'right' and 'wrong' are not the same lenght")
    
    return [100*(right[i]/(right[i]+wrong[i])) for i in range(len(right))]

def lr_exp_decay(epoch, lr):
    k = 0.1
    return lr * math.exp(-k*epoch)

def validate_monitor(monitor, best_accuracy, accuracy, best_loss, loss):
    if monitor == 'val_loss':
        return loss < best_loss or best_loss == -1
    elif monitor == 'val_acc':
        return accuracy > best_accuracy or best_accuracy == -1
    else:
        raise TypeError(f'{monitor} is not a valid evaluation monitor')

def calc_accuracy(right, wrong):
    return right / (right + wrong)

def average(tim):
    return sum(tim) / len(tim)

def fit_model(model, train_img, train_lab, val_img, val_lab, filter_names, apply_noise_method, monitor='val_loss',
            delta_value=None, patience=10, epochs=100, restore_weights=False, augmentation=False
            ):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    history_monitor = []
    batch_size = 32
    
    best_epoch = -1
    best_loss = -1
    best_accuracy = -1
    best_model = copy(model)
    
    current_learning_rate = 0.001
    
    right, wrong = [], []
    
    done = epochs
    progress = trange(done, desc='epoch stuff', leave=True)
    loss = 0
    
    times = []
    tik = time.perf_counter()
    for epoch in progress:
        progress.set_description(f"E = {epoch}, LR = {current_learning_rate}, LOSS = {loss}")
        progress.refresh()
        
        if epoch != 0:
            current_learning_rate = lr_exp_decay(epoch, current_learning_rate)
            K.set_value(model.optimizer.learning_rate, current_learning_rate)
        
        for (xb, yb) in get_batch(train_img, train_lab, batch_size, filter_names, apply_noise_method, augmentation=augmentation):
            
            xb = np.array(xb)
            yb = np.array(yb)
            try:
                _ = model.train_on_batch(tf.convert_to_tensor(xb) , tf.convert_to_tensor(yb))
            except Exception as e:
                print(f"ERROR: {e}")
                raise Exception
        
        img_predict = []
        img_true = []
        
        right.append(0)
        wrong.append(0)
        
        for xb, yb in get_batch(val_img, val_lab, batch_size, filter_names, apply_noise_method, augmentation=augmentation):
            for i in range(len(xb)):
                prediction = make_prediction(model, xb[i], (52, 52, 3)).numpy()[0] #TODO resolution is hard coded. pls fix
                img_predict.append(prediction)
                img_true.append(int(yb[i]))
                
                predicted_label = np.argmax(prediction)
                
                if predicted_label == int(yb[i]):
                    right[-1] += 1
                else:
                    wrong[-1] += 1

        loss = scce(img_true, img_predict).numpy()
        history_monitor.append(loss)
        
        current_accuracy = calc_accuracy(right[-1], wrong[-1])
        if validate_monitor(monitor, best_accuracy, current_accuracy, best_loss, loss) and should_early_stop:
        # if loss < best_loss or best_loss == -1 and should_early_stop:
            best_epoch = epoch
            best_loss = loss
            best_accuracy = calc_accuracy(right[-1], wrong[-1])
            best_model = copy(model)
        elif should_early_stop(best_epoch, epoch, patience) and should_early_stop:
            return best_model, history_monitor, sum_accuracy(right, wrong)


    tok = time.perf_counter()
    print(f"EPOCH TIME: {tok-tik}")
    
    return_acuracy = sum_accuracy(right, wrong)
    
    if should_early_stop:
        return best_model, history_monitor, return_acuracy
    else:
        return model, history_monitor, return_acuracy
    
    
    # return best_model, history_monitor, return_acuracy if should_early_stop else model, history_monitor, return_acuracy
