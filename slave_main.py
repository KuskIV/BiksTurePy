
from global_paths import get_paths, get_h5_test, get_h5_train, get_h5_test_noise, get_h5_train_noise, get_h5_train_homo, get_h5_test_homo
from phase_one.main_phase_one import ex_two_eval_norm, ex_one, run_experiment_one
from phase_two.main_phase_two import ex_two_eval_noise
from phase_one.find_ideal_model import get_satina_gains_model_norm_object_list, get_satina_gains_model_object_list
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

def load_dehaze_filters()->dict:
    F = premade_single_filter('fog_dehaze')
    R = premade_single_filter('rain_dehaze')
    S = premade_single_filter('snow_dehaze')
    D = premade_single_filter('day_dehaze')
    N = premade_single_filter('night_dehaze')
    
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

def load_lobster_level_filters_fog()->dict:
    F1 = premade_single_filter('mod_fog0.1')
    F2 = premade_single_filter('mod_fog0.2')
    F3 = premade_single_filter('mod_fog0.3')
    F4 = premade_single_filter('mod_fog0.4')
    F5 = premade_single_filter('mod_fog0.5')
    F6 = premade_single_filter('mod_fog0.6')
    F7 = premade_single_filter('mod_fog0.7')
    F8 = premade_single_filter('mod_fog0.8')
    F9 = premade_single_filter('mod_fog0.9')
    F10 = premade_single_filter('mod_fog1.0')
    
    dict = [{'fog1':F1},{'fog2':F2},{'fog3':F3},
            {'fog4':F4},{'fog5':F5},{'fog6':F6},
            {'fog7':F7},{'fog8':F8},{'fog9':F9},
            {'fog10':F10}
        ]
    return dict

def load_lobster_level_filters_night()->dict:
    F1 = premade_single_filter('mod_night0.1')
    F2 = premade_single_filter('mod_night0.2')
    F3 = premade_single_filter('mod_night0.3')
    F4 = premade_single_filter('mod_night0.4')
    F5 = premade_single_filter('mod_night0.5')
    F6 = premade_single_filter('mod_night0.6')
    F7 = premade_single_filter('mod_night0.7')
    F8 = premade_single_filter('mod_night0.8')
    F9 = premade_single_filter('mod_night0.9')
    F10 = premade_single_filter('mod_night1.0')
    
    dict = [{'night1':F1},{'night2':F2},{'night3':F3},
            {'night4':F4},{'night5':F5},{'night6':F6},
            {'night7':F7},{'night8':F8},{'night9':F9},
            {'night10':F10}
        ]
    return dict

def load_lobster_level_filters_rain()->dict:
    F1 = premade_single_filter('mod_rain0.1')
    F2 = premade_single_filter('mod_rain0.2')
    F3 = premade_single_filter('mod_rain0.3')
    F4 = premade_single_filter('mod_rain0.4')
    F5 = premade_single_filter('mod_rain0.5')
    F6 = premade_single_filter('mod_rain0.6')
    F7 = premade_single_filter('mod_rain0.7')
    F8 = premade_single_filter('mod_rain0.8')
    F9 = premade_single_filter('mod_rain0.9')
    F10 = premade_single_filter('mod_rain1.0')
    
    dict = [{'rain1':F1},{'rain2':F2},{'rain3':F3},
            {'rain4':F4},{'rain5':F5},{'rain6':F6},
            {'rain7':F7},{'rain8':F8},{'rain9':F9},
            {'rain10':F10}
        ]
    return dict

def load_lobster_level_filters_snow()->dict:
    F1 = premade_single_filter('mod_snow0.1')
    F2 = premade_single_filter('mod_snow0.2')
    F3 = premade_single_filter('mod_snow0.3')
    F4 = premade_single_filter('mod_snow0.4')
    F5 = premade_single_filter('mod_snow0.5')
    F6 = premade_single_filter('mod_snow0.6')
    F7 = premade_single_filter('mod_snow0.7')
    F8 = premade_single_filter('mod_snow0.8')
    F9 = premade_single_filter('mod_snow0.9')
    F10 = premade_single_filter('mod_snow1.0')
    
    dict = [{'snow1':F1},{'snow2':F2},{'snow3':F3},
            {'snow4':F4},{'snow5':F5},{'snow6':F6},
            {'snow7':F7},{'snow8':F8},{'snow9':F9},
            {'snow10':F10}
        ]
    return dict

def create_lobster_dir(base_ex, lobster1, lobster2, index):
    create_dir(f"{get_paths('phase_two_csv')}/{lobster1}")
    create_dir(f"{get_paths('phase_two_csv')}/{lobster2}")

def create_dirs(base_ex, base_result, base_big_lobster, index):
    create_dir(f"{get_paths('phase_one_csv')}/{index}")
    create_dir(f"{get_paths('phase_two_csv')}/{index}")
    
    create_dir(f"{get_paths('phase_one_csv')}/{base_ex}")
    create_dir(f"{get_paths('phase_two_csv')}/{base_ex}")
    
    create_dir(f"{get_paths('phase_one_csv')}/{base_result}")
    create_dir(f"{get_paths('phase_two_csv')}/{base_result}")

    create_dir(f"{get_paths('phase_two_csv')}/{base_big_lobster}")
    
    # create_dir(f"{get_paths('phase_two_csv')}/{base_big_lobster_level}")

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

def get_dehaze_path():
    return [get_paths('satina_median_dehaze'), get_paths('satina_avg_dehaze'), get_paths('satina_mode_dehaze')]

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

def create_paths_for_lobster_noise_level(base_path, folder_name):
    path_to_create1 = f"{get_paths('phase_two_csv')}/{base_path}"
    path_to_create2 = f"{get_paths('phase_two_csv')}/{folder_name}"
    create_dir(path_to_create1)
    create_dir(path_to_create2)

def one_lobster_noise_level(noise_name, base_path, test_path, data_to_test_on, ideal_path, filter_method, ideal_and_lobster_on_one_model, folder_extension):
    errors = ''
    folder_name = ''
    
    try:
        modified_lobster_level_folder = f"experiment_two_big_lobster_{folder_extension}level{noise_name}"
        modified_base_path = f"{base_path}{folder_extension}"
        ex_folder = get_ex_folder(modified_lobster_level_folder, modified_base_path)
        create_paths_for_lobster_noise_level(modified_base_path, ex_folder)
        folder_name = modified_base_path
        introduce_experiment(modified_lobster_level_folder)
        ex_two_eval_noise(test_path, ex_folder, get_models=get_satina_gains_model_norm_object_list, 
                        data_to_test_on=data_to_test_on, model_paths=ideal_path, filter_method=filter_method, 
                        run_on_one_model=ideal_and_lobster_on_one_model)
    except Exception as e:
        print(f"ERROR IN EXPERIMENT 'TRAIN ON IDEAL LOBSTER {noise_name}'")
        e = sys.exc_info()
        print(e)
        errors = e
    return errors, folder_name

def get_errors_and_folders(errors, folders):
    error_list = [e for e in errors if e != '']
    folder_list = [f for f in folders if f != '']
    
    return error_list, folder_list

def lobster_noise_level(noise_name, data_to_test_on, base_path, test_path, ideal_path, ideal_noise_path, filter_method, ideal_and_lobster_on_one_model):
    e1, f1 = one_lobster_noise_level(noise_name, base_path, test_path, data_to_test_on, ideal_path, filter_method, ideal_and_lobster_on_one_model, 'ideal')
    e2, f2 = one_lobster_noise_level(noise_name, base_path, test_path, data_to_test_on, ideal_noise_path, filter_method, ideal_and_lobster_on_one_model, 'noise')
    
    return get_errors_and_folders([e1, e2], [f1, f2])

def extend_errors(errors, errors_to_append):
    for e in errors_to_append:
        if len(e) != 0:
            errors.extend(e)

def get_method_for_models(model_type):
    if model_type == 'norm':
        return get_satina_gains_model_norm_object_list
    elif model_type == 'base':
        return get_satina_gains_model_object_list
    else:
        raise TypeError(f"{model_type} is not a valid option. Should be either 'norm' or 'base'")

def run_default_experiment(folder_name:str, base_folder:str, test_path:str, train_path:str, data_to_test_on:int, 
                        model_types:str, filter_method:list=None, condition=True, model_paths=None, train_model=True, run_on_one_model=False, two_test_path=None) -> str:
    
    errors = ''
    
    get_models_method = get_method_for_models(model_types)
    
    if two_test_path == None:
        two_test_path = test_path
    
    try:
        if condition:
            ex_folder = get_ex_folder(folder_name, base_folder)
            introduce_experiment(folder_name)
            if train_model:
                run_experiment_one(1, train_path, test_path, get_models_method, 
                    epochs_end=100, folder_extension=ex_folder, data_to_test_on=data_to_test_on, 
                    model_paths=model_paths, run_on_one_model=run_on_one_model)
                
                
                # run_experiment_one(test_path, train_path, get_models_method, folder_extension=ex_folder, 
                #                 data_to_test_on=data_to_test_on, model_paths=None, run_on_one_model=run_on_one_model)
            
            ex_two_eval_noise(two_test_path, ex_folder, get_models=get_models_method, data_to_test_on=data_to_test_on, 
                            model_paths=model_paths, filter_method=filter_method, run_on_one_model=run_on_one_model)
        else:
            print("---\nWARNING: experiment lobster_noise will not run since an error occured when training the model\n---")
    except Exception as e:
        print(f"ERROR IN EXPERIMENT '{folder_name}'")
        e = sys.exc_info()
        print(e)
        errors = e
    
    return errors

def run_biksture(index, data_to_test_on, run_base_experiments=True, run_ideal_experiments=True, 
                run_lobster_experiments=True, run_lobster_level_experiments=True):
    test_path = get_h5_test()
    train_path = get_h5_train()
    
    noise_test_path = get_h5_test_noise()
    noise_train_path = get_h5_train_noise()
    
    homo_test_path = get_h5_test_homo()
    homo_train_path = get_h5_train_homo()
    
    dehaze_test_path = get_paths('h5_test_dehaze')
    dehaze_train_path = get_paths('h5_train_dehaze')
    
    ideal_noise_test_path = get_paths('h5_test_ideal_noise')
    ideal_noise_train_path = get_paths('h5_train_ideal_noise')
    
    noise_paths = get_noise_paths()
    homo_path = get_homo_paths()
    ideal_path = get_ideal_paths()
    ideal_noise_path = get_ideal_noise_paths()
    dehaze_path = get_dehaze_path()
    
    ideal_noise_worked = False
    ideal_worked = False
    
    exclude_folders = []
    
    ideal_and_lobster_on_one_model = True
    
    base_ex = f"{index}/experiment_two_data"
    base_result = f"{index}/experiment_two_result"
    base_big_lobster = f"{index}/experiment_big_lobster"
    base_big_lobster_level = f"{index}/experiment_big_lobster_level"

    create_dirs(base_ex, base_result, base_big_lobster, index)
    
    errors = []

    if run_base_experiments:
        # # Baseline experiment
        # baseline_folder = "experiment_baseline"
        # run_default_experiment(baseline_folder, base_ex, test_path, train_path, data_to_test_on, 'base', 
        #                 filter_method=None, condition=True, model_paths=None, train_model=True, run_on_one_model=False)
        
        # # Normalized experiment
        # norm_folder = "experiment_two_eval_norm"
        # run_default_experiment(norm_folder, base_ex, test_path, train_path, data_to_test_on, 'norm', 
        #                 filter_method=None, condition=True, model_paths=None, train_model=True, run_on_one_model=False)
        
        # # Homomorpic experiment
        # homo_folder = "experiment_two_eval_homo"
        # run_default_experiment(homo_folder, base_ex, homo_test_path, homo_train_path, data_to_test_on, 'base', 
        #                 filter_method=load_homo_filters, condition=True, model_paths=homo_path, train_model=True, run_on_one_model=False)
        
        # # Dehaze experiment
        # dehaze_folder = "experiment_two_eval_dehaze"
        # run_default_experiment(dehaze_folder, base_ex, dehaze_test_path, dehaze_train_path, data_to_test_on, 'base', 
        #                 filter_method=load_dehaze_filters, condition=True, model_paths=dehaze_path, train_model=True, run_on_one_model=False)
        
        # # Noise experiment
        # noise_folder = "experiment_two_eval_noise"
        # run_default_experiment(noise_folder, base_ex, noise_test_path, noise_train_path, data_to_test_on, 'base', filter_method=None, 
        #                 condition=True, model_paths=noise_paths, train_model=True, run_on_one_model=False, two_test_path=test_path)
        

        run_default_experiment(folder_name, base_folder, test_path, train_path, get_models_method, data_to_test_on, 
                        model_types, filter_method=None, condition=True, model_paths=None, train_model=True, run_on_one_model=False)
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
            ex_one(homo_test_path, homo_train_path, folder_extension=ex_folder, data_to_test_on=data_to_test_on, model_paths=homo_path)
            ex_two_eval_noise(homo_test_path, ex_folder, data_to_test_on=data_to_test_on, model_paths=homo_path, filter_method=load_homo_filters)
        except:
            print("ERROR IN EXPERIMENT 'TRAIN ON HOMO'")
            e = sys.exc_info()
            print(e)
            errors.append(e)
        

        
        try:
            dehaze_folder = "experiment_two_eval_dehaze"
            ex_folder = get_ex_folder(dehaze_folder, base_ex)
            introduce_experiment(dehaze_folder)
            ex_one(dehaze_test_path, dehaze_train_path, folder_extension=ex_folder, data_to_test_on=data_to_test_on, model_paths=dehaze_path)
            ex_two_eval_noise(dehaze_test_path, ex_folder, data_to_test_on=data_to_test_on, filter_method=load_dehaze_filters, model_paths=dehaze_path)
        except:
            print("ERROR IN EXPERIMENT 'TRAIN ON DEHAZE'")
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
            ex_two_eval_norm(homo_test_path, homo_train_path, folder_extension=ex_folder, data_to_test_on=data_to_test_on, model_paths=ideal_path, run_on_one_model=ideal_and_lobster_on_one_model)
            ideal_worked = True
            ex_two_eval_noise(homo_test_path, ex_folder, get_models=get_satina_gains_model_norm_object_list, data_to_test_on=data_to_test_on, model_paths=ideal_path, filter_method=load_homo_filters, run_on_one_model=ideal_and_lobster_on_one_model)
        except Exception as e:
            print("ERROR IN EXPERIMENT 'TRAIN ON IDEAL'")
            e = sys.exc_info()
            print(e)
            errors.append(e)
            
        try:
            ideal_noise_folder = "experiment_two_eval_idealnoise"
            ex_folder = get_ex_folder(ideal_noise_folder, base_result)
            introduce_experiment(ideal_noise_folder)
            ex_two_eval_norm(ideal_noise_test_path, ideal_noise_train_path, folder_extension=ex_folder, data_to_test_on=data_to_test_on, model_paths=ideal_noise_path, run_on_one_model=ideal_and_lobster_on_one_model)
            ideal_noise_worked = True
            ex_two_eval_noise(homo_test_path, ex_folder, get_models=get_satina_gains_model_norm_object_list, data_to_test_on=data_to_test_on, model_paths=ideal_noise_path, filter_method=load_homo_filters, run_on_one_model=ideal_and_lobster_on_one_model)
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
                ex_two_eval_noise(homo_test_path, ex_folder, get_models=get_satina_gains_model_norm_object_list, data_to_test_on=data_to_test_on, model_paths=ideal_path, filter_method=load_lobster_filters, run_on_one_model=ideal_and_lobster_on_one_model)
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
                ex_two_eval_noise(homo_test_path, ex_folder, get_models=get_satina_gains_model_norm_object_list, data_to_test_on=data_to_test_on, model_paths=ideal_noise_path, filter_method=load_lobster_filters, run_on_one_model=ideal_and_lobster_on_one_model)
            else:
                print("---\nWARNING: experiment lobster_noise will not run since an error occured when training the model\n---")
        except Exception as e:
            print("ERROR IN EXPERIMENT 'TRAIN ON NOISE LOBSTER'")
            e = sys.exc_info()
            print(e)
            errors.append(e)
    
    if run_lobster_level_experiments:
        e1, _ = lobster_noise_level('fog', data_to_test_on, base_big_lobster_level, homo_test_path, ideal_path, ideal_noise_path, load_lobster_level_filters_fog, ideal_and_lobster_on_one_model)
        e2, _ = lobster_noise_level('night', data_to_test_on, base_big_lobster_level, homo_test_path, ideal_path, ideal_noise_path, load_lobster_level_filters_night, ideal_and_lobster_on_one_model)
        e3, _ = lobster_noise_level('rain', data_to_test_on, base_big_lobster_level, homo_test_path, ideal_path, ideal_noise_path, load_lobster_level_filters_rain, ideal_and_lobster_on_one_model)
        e4, exclude_folders = lobster_noise_level('snow', data_to_test_on, base_big_lobster_level, homo_test_path, ideal_path, ideal_noise_path, load_lobster_level_filters_snow, ideal_and_lobster_on_one_model)
        extend_errors(errors, [e1, e2, e3, e4])

    try:
        sum_merged_files(f'phase_two/csv_output/{index}', exclude_folders)
    except Exception as e:
        print(f"ERROR: {e}")
    
    if len(errors) != 0:
        time_str = time.strftime("%Y%m%d-%H%M%S")
        save_path = f"error_messages/output_error_{time_str}.txt"
        
        print("---------------------------")
        print(f"During execution, errors occured in {len(errors)} experiments. These errors can be found in the following txt documtn:\n{save_path}")
        print("---------------------------")
        
        with open(save_path, 'w') as output:
            output.write(str(errors))
