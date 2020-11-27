
from global_paths import get_paths, get_h5_test, get_h5_train, get_h5_test_noise, get_h5_train_noise, get_h5_train_homo, get_h5_test_homo
from phase_one.main_phase_one import ex_two_eval_norm, ex_one
from phase_two.main_phase_two import ex_two_eval_noise
from phase_one.find_ideal_model import get_satina_gains_model_norm_object_list
from phase_two.sum_phase_two import sum_merged_files
from Noise_Generators.noise_main import premade_single_filter
import sys
import time
import math
from PIL import Image
import numpy as np
import os
import shutil

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def load_homo_filters()->dict:
    F = premade_single_filter('foghomo')
    R = premade_single_filter('rainhomo')
    S = premade_single_filter('snowhomo')
    D = premade_single_filter('dayhomo')
    N = premade_single_filter('nighthomo')
    
    #dict = {'fog':F,'rain':R,'snow':S,'day':D,'night':N}
    dict = [{'fog':F}, {'night':N},{'rain':R},{'snow':S},{'day':D},{'night':N}]
    return dict

def load_lobster_filters()->dict:
    FN = premade_single_filter('fog_night')
    FS = premade_single_filter('fog_snow')
    FR = premade_single_filter('fog_rain')
    RN = premade_single_filter('rain_night')
    SN = premade_single_filter('snow_night')
    
    dict = [{'fognight':FN}, {'fogsnow':FS},{'fograin':FR},{'rainnight':RN},{'snownight':SN}]
    return dict

def load_lobster_level_filters()->dict:
    FL = premade_single_filter('fog_mild')
    FM = premade_single_filter('fog_medium')
    FH = premade_single_filter('fog_heavy')
    RL = premade_single_filter('rain_mild')
    RM = premade_single_filter('rain_medium')
    RH = premade_single_filter('rain_heavy')
    SL = premade_single_filter('snow_mild')
    SM = premade_single_filter('snow_medium')
    SH = premade_single_filter('snow_heavy')
    NL = premade_single_filter('night_mild')
    NM = premade_single_filter('night_medium')
    NH = premade_single_filter('night_heavy')
    
    dict = [{'fogmild':FL},{'fogmedium':FM},{'fogheavy':FH},
            {'rainmild':RL},{'rainmedium':RM},{'rainheavy':RH},
            {'snowmild':SL},{'snowmedium':SM},{'snowheavy':SH},
            {'nightmild':NL},{'nightmedium':NM},{'nightheavy':NH}
        ]
    return dict

def create_dirs(base_ex, base_result, base_big_lobster, base_big_lobster_level):
    create_dir(f"{get_paths('phase_one_csv')}/{base_ex}")
    create_dir(f"{get_paths('phase_two_csv')}/{base_ex}")
    
    create_dir(f"{get_paths('phase_one_csv')}/{base_result}")
    create_dir(f"{get_paths('phase_two_csv')}/{base_result}")

    create_dir(f"{get_paths('phase_two_csv')}/{base_big_lobster}")
    
    create_dir(f"{get_paths('phase_two_csv')}/{base_big_lobster_level}")

def apply_noise_evenly(img_batch, noises, batch_size):
    global_idx = 0
    aug_bs = batch_size // len(noises)
    
    for i, noise in enumerate(noises):
        for img in img_batch[i*aug_bs:i+1*aug_bs]:
            img_batch[global_idx] = noise + img
            global_idx += 1

def get_noise_evenly_tuple():
    return apply_noise_evenly, premade_single_filter('NOT IMPLEMETED')

def apply_got_twenty(img_batch, noise, batch_size):
    # global_idx = 0
    aug_bs = math.ceil((batch_size / 100) * 20)
    img_batch[:aug_bs] = noise * img_batch[:aug_bs]
    return img_batch
    # for img in img_batch[:aug_bs]:
    #     img_batch[global_idx] = noise + img
    #     global_idx += 1

def get_fog_twenty_tuple():
    return apply_got_twenty, premade_single_filter('night'), True

def get_noise_paths():
    return [get_paths('satina_median_noise'), get_paths('satina_avg_noise'), get_paths('satina_mode_noise')]
    # return [get_satina_model_median_path_noise(), get_satina_model_avg_path_noise(), get_satina_model_mode_path_noise()]

def get_homo_paths():
    return [get_paths('satina_median_homo'), get_paths('satina_avg_homo'), get_paths('satina_mode_homo')]
    # return [get_satina_model_median_path_homo(), get_satina_model_avg_path_homo(), get_satina_model_mode_path_homo()]

def get_ideal_paths():
    return [get_paths('satina_median_ideal'), get_paths('satina_avg_ideal'), get_paths('satina_mode_ideal')]
    # return [get_satina_model_median_path_ideal(), get_satina_model_avg_path_ideal(), get_satina_model_mode_path_ideal()]

def get_ideal_noise_paths():
    return [get_paths('satina_median_idealnoise'), get_paths('satina_avg_idealnoise'), get_paths('satina_mode_idealnoise')]
    # return [get_satina_model_median_path_ideal_noise(), get_satina_model_avg_path_ideal_noise(), get_satina_model_mode_path_ideal_noise()]

def introduce_experiment(folder_name):
    try:
        print("-----------------------")
        print(f"A new experiment is about to start:")
        print(f"The experiment is: {folder_name}")
        print("-----------------------")
    except TypeError:
        print("jeppe pls fuck off, dont exploit my methods you monster")

def get_ex_folder(baseline_folder, base):
    return f"{base}/{baseline_folder}"

if __name__ == "__main__":
    test_path = get_h5_test()
    train_path = get_h5_train()
    
    noise_test_path = get_h5_test_noise()
    noise_train_path = get_h5_train_noise()
    
    homo_test_path = get_h5_test_homo()
    homo_train_path = get_h5_train_homo()
    
    ideal_noise_test_path = get_paths('h5_test_ideal_noise')
    ideal_noise_train_path = get_paths('h5_train_ideal_noise')
    
    noise_paths = get_noise_paths()
    homo_path = get_homo_paths()
    ideal_path = get_ideal_paths()
    ideal_noise_path = get_ideal_noise_paths()
    
    run_base_experiments = False
    run_ideal_experiments = False
    run_lobster_experiments = False
    run_lobster_level_experiments = True
    
    ideal_noise_worked = False
    ideal_worked = False
    
    base_ex = "experiment_two_data"
    base_result = "experiment_two_result"
    base_big_lobster = "experiment_big_lobster"
    base_big_lobster_level = "experiment_big_lobster_level"
    
    create_dirs(base_ex, base_result, base_big_lobster, base_big_lobster_level)

    data_to_test_on = 1
    
    errors = []
    # TODO train the models 5 times and take average of the fitdata

    if run_base_experiments:
        try:
            baseline_folder = "experiment_baseline"
            ex_folder = get_ex_folder(baseline_folder, base_ex)
            introduce_experiment(baseline_folder)
            ex_one(test_path, train_path, folder_extension=ex_folder, data_to_test_on=data_to_test_on)
            ex_two_eval_noise(test_path, ex_folder, data_to_test_on=data_to_test_on)
        except:
            print("ERROR IN EXPERIMENT 'TRAIN ON BASELINE'")
            e = sys.exc_info()
            print(e)
            errors.append(e)

        try:
            norm_folder = "experiment_two_eval_norm"
            ex_folder = get_ex_folder(norm_folder, base_ex)
            introduce_experiment(norm_folder)
            ex_two_eval_norm(test_path, train_path, folder_extension=ex_folder, data_to_test_on=data_to_test_on)
            ex_two_eval_noise(test_path, ex_folder, get_models=get_satina_gains_model_norm_object_list, data_to_test_on=data_to_test_on)
        except:
            print("ERROR IN EXPERIMENT 'TRAIN ON NORM'")
            e = sys.exc_info()
            print(e)
            errors.append(e)
        
        try:
            homo_folder = "experiment_two_eval_homo"
            ex_folder = get_ex_folder(homo_folder, base_ex)
            introduce_experiment(homo_folder)
            # ex_one(test_path, train_path, folder_extension=homo_folder, data_to_test_on=data_to_test_on, model_paths=homo_path, noise_tuple=get_fog_twenty_tuple())
            ex_one(homo_test_path, homo_train_path, folder_extension=ex_folder, data_to_test_on=data_to_test_on, model_paths=homo_path, noise_tuple=get_fog_twenty_tuple())
            ex_two_eval_noise(homo_test_path, ex_folder, data_to_test_on=data_to_test_on, model_paths=homo_path, filter_method=load_homo_filters)
        except:
            print("ERROR IN EXPERIMENT 'TRAIN ON HOMO'")
            e = sys.exc_info()
            print(e)
            errors.append(e)
        
        try:
            noise_folder = "experiment_two_eval_noise"
            ex_folder = get_ex_folder(noise_folder, base_ex)
            introduce_experiment(noise_folder)
            ex_one(noise_test_path, noise_train_path, folder_extension=ex_folder, model_paths=noise_paths, data_to_test_on=data_to_test_on)
            ex_two_eval_noise(test_path, ex_folder, model_paths=noise_paths, data_to_test_on=data_to_test_on)
        except:
            print("ERROR IN EXPERIMENT 'TRAIN ON NOISE'")
            e = sys.exc_info()
            print(e)
            errors.append(e)

    if run_ideal_experiments:
        try:
            ideal_folder = "experiment_two_eval_ideal"
            ex_folder = get_ex_folder(ideal_folder, base_result)
            introduce_experiment(ideal_folder)
            ex_two_eval_norm(homo_test_path, homo_train_path, folder_extension=ex_folder, data_to_test_on=data_to_test_on, model_paths=ideal_path)
            ex_two_eval_noise(homo_test_path, ex_folder, get_models=get_satina_gains_model_norm_object_list, data_to_test_on=data_to_test_on, model_paths=ideal_path, filter_method=load_homo_filters)
            ideal_worked = True
        except Exception as e:
            print("ERROR IN EXPERIMENT 'TRAIN ON IDEAL'")
            e = sys.exc_info()
            print(e)
            errors.append(e)
            
        try:
            ideal_noise_folder = "experiment_two_eval_idealnoise"
            ex_folder = get_ex_folder(ideal_noise_folder, base_result)
            introduce_experiment(ideal_noise_folder)
            ex_two_eval_norm(ideal_noise_test_path, ideal_noise_train_path, folder_extension=ex_folder, data_to_test_on=data_to_test_on, model_paths=ideal_noise_path)
            ex_two_eval_noise(homo_test_path, ex_folder, get_models=get_satina_gains_model_norm_object_list, data_to_test_on=data_to_test_on, model_paths=ideal_noise_path, filter_method=load_homo_filters)
            ideal_noise_worked = True
        except Exception as e:
            print("ERROR IN EXPERIMENT 'TRAIN ON IDEAL'")
            e = sys.exc_info()
            print(e)
            errors.append(e)

    if run_lobster_experiments:
        try:
            if ideal_worked or not run_ideal_experiments:
                ideal_lobster_folder = "experiment_two_big_lobster_ideal"
                ex_folder = get_ex_folder(ideal_lobster_folder, base_big_lobster)
                introduce_experiment(ideal_lobster_folder)
                ex_two_eval_noise(homo_test_path, ex_folder, get_models=get_satina_gains_model_norm_object_list, data_to_test_on=data_to_test_on, model_paths=ideal_path, filter_method=load_lobster_filters)
            else:
                print("---\nWARNING: experiment lobster_ideal will not run since an error occured when training the model\n---")
        except Exception as e:
            print("ERROR IN EXPERIMENT 'TRAIN ON IDEAL LOBSTER'")
            e = sys.exc_info()
            print(e)
            errors.append(e)
        
        try:
            if ideal_worked or not run_ideal_experiments:
                noise_lobster_folder = "experiment_two_big_lobster_noise"
                ex_folder = get_ex_folder(noise_lobster_folder, base_big_lobster)
                introduce_experiment(noise_lobster_folder)
                ex_two_eval_noise(homo_test_path, ex_folder, get_models=get_satina_gains_model_norm_object_list, data_to_test_on=data_to_test_on, model_paths=ideal_noise_path, filter_method=load_lobster_filters)
            else:
                print("---\nWARNING: experiment lobster_noise will not run since an error occured when training the model\n---")
        except Exception as e:
            print("ERROR IN EXPERIMENT 'TRAIN ON NOISE LOBSTER'")
            e = sys.exc_info()
            print(e)
            errors.append(e)
    
    
    if run_lobster_level_experiments:
        try:
            if ideal_worked or not run_ideal_experiments:
                ideal_lobster_level_folder = "experiment_two_big_lobster_ideallevel"
                ex_folder = get_ex_folder(ideal_lobster_level_folder, base_big_lobster_level)
                introduce_experiment(ideal_lobster_level_folder)
                ex_two_eval_noise(homo_test_path, ex_folder, get_models=get_satina_gains_model_norm_object_list, data_to_test_on=data_to_test_on, model_paths=ideal_path, filter_method=load_lobster_level_filters)
            else:
                print("---\nWARNING: experiment lobster_ideal will not run since an error occured when training the model\n---")
        except Exception as e:
            print("ERROR IN EXPERIMENT 'TRAIN ON IDEAL LOBSTER'")
            e = sys.exc_info()
            print(e)
            errors.append(e)
        
        try:
            if ideal_worked or not run_ideal_experiments:
                noise_lobster_level_folder = "experiment_two_big_lobster_noiselevel"
                ex_folder = get_ex_folder(noise_lobster_level_folder, base_big_lobster_level)
                introduce_experiment(noise_lobster_level_folder)
                ex_two_eval_noise(homo_test_path, ex_folder, get_models=get_satina_gains_model_norm_object_list, data_to_test_on=data_to_test_on, model_paths=ideal_noise_path, filter_method=load_lobster_level_filters)
            else:
                print("---\nWARNING: experiment lobster_noise will not run since an error occured when training the model\n---")
        except Exception as e:
            print("ERROR IN EXPERIMENT 'TRAIN ON NOISE LOBSTER'")
            e = sys.exc_info()
            print(e)
            errors.append(e)

    sum_merged_files('phase_two/csv_output')

    if len(errors) != 0:
        time_str = time.strftime("%Y%m%d-%H%M%S")
        save_path = f"error_messages/output_error_{time_str}.txt"
        
        print("---------------------------")
        print(f"During execution, errors occured in {len(errors)} experiments. These errors can be found in the following txt documtn:\n{save_path}")
        print("---------------------------")
        
        with open(save_path, 'w') as output:
            output.write(str(errors))
