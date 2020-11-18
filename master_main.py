
from global_paths import get_h5_test, get_h5_train, get_h5_test_noise, get_h5_train_noise, get_satina_model_mode_path_noise, get_satina_model_median_path_noise, get_satina_model_avg_path_noise
from phase_one.main_phase_one import ex_two_eval_norm, ex_one
from phase_two.main_phase_two import ex_two_eval_noise
from phase_one.find_ideal_model import get_satina_gains_model_norm_object_list
from phase_two.sum_phase_two import sum_merged_files
import sys
import time

def get_noise_paths():
    return [get_satina_model_median_path_noise(), get_satina_model_avg_path_noise(), get_satina_model_mode_path_noise()]

def introduce_experiment(folder_name):
    try:
        print("-----------------------")
        print(f"A new experiment is about to start:")
        print(f"The experiment is: {folder_name}")
        print("-----------------------")
    except TypeError:
        print("jeppe pls fuck off, dont exploit my methods you monster")

if __name__ == "__main__":
    test_path = get_h5_test()
    train_path = get_h5_train()
    
    noise_test_path = get_h5_test_noise()
    noise_train_path = get_h5_train_noise()
    
    noise_paths = get_noise_paths()
    
    data_to_test_on = 100
    
    errors = []

    try:
        baseline_folder = "experiment_baseline"
        introduce_experiment(baseline_folder)
        ex_one(test_path, train_path, folder_extension=baseline_folder, data_to_test_on=data_to_test_on)
        ex_two_eval_noise(test_path, baseline_folder, data_to_test_on=data_to_test_on)
    except:
        print("ERROR IN EXPERIMENT 'TRAIN ON BASELINE'")
        e = sys.exc_info()
        print(e)
        errors.append(e)

    try:
        norm_folder = "experiment_two_eval_norm"
        introduce_experiment(norm_folder)
        ex_two_eval_norm(test_path, train_path, folder_extension=norm_folder, data_to_test_on=data_to_test_on)
        ex_two_eval_noise(test_path, norm_folder, get_models=get_satina_gains_model_norm_object_list, data_to_test_on=data_to_test_on)
    except:
        print("ERROR IN EXPERIMENT 'TRAIN ON NORM'")
        e = sys.exc_info()
        print(e)
        errors.append(e)
    
    try:
        noise_folder = "experiment_two_eval_noise"
        introduce_experiment(noise_folder)
        ex_one(noise_test_path, noise_train_path, folder_extension=noise_folder, model_paths=noise_paths, data_to_test_on=data_to_test_on)
        ex_two_eval_noise(test_path, noise_folder, model_paths=noise_paths, data_to_test_on=data_to_test_on)
    except:
        print("ERROR IN EXPERIMENT 'TRAIN ON NOISE'")
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
