
from global_paths import get_h5_test, get_h5_train, get_h5_test_noise, get_h5_train_noise, get_satina_model_mode_path_noise, get_satina_model_median_path_noise, get_satina_model_avg_path_noise
from phase_one.main_phase_one import ex_two_eval_norm, ex_one
from phase_two.main_phase_two import ex_two_eval_noise
from phase_one.find_ideal_model import get_satina_gains_model_norm_object_list
from phase_two.sum_phase_two import sum_merged_files
import sys
import time

if __name__ == "__main__":
    test_path = get_h5_test()
    train_path = get_h5_train()
    
    noise_paths = [get_satina_model_median_path_noise(), get_satina_model_avg_path_noise(), get_satina_model_mode_path_noise()]

    noise_test_path = get_h5_test_noise()
    noise_train_path = get_h5_train_noise()
    
    data_to_test_on = 50
    
    errors = []

    # try:
    #     ex_one(test_path, train_path, folder_extension="experiment_one", data_to_test_on=data_to_test_on)
    #     ex_two_eval_noise(test_path, 'experiment_two_eval_baseline', data_to_test_on=data_to_test_on)
    # except:
    #     print("ERROR IN EXPERIMENT 'TRAIN ON BASELINE'")
    #     e = sys.exc_info()
    #     print(e)
    #     errors.append(e)

    # try:
    #     ex_two_eval_norm(test_path, train_path, folder_extension="experiment_two_eval_norm", data_to_test_on=data_to_test_on)
    #     ex_two_eval_noise(test_path, 'experiment_two_eval_norm', get_models=get_satina_gains_model_norm_object_list, data_to_test_on=data_to_test_on)
    # except:
    #     print("ERROR IN EXPERIMENT 'TRAIN ON NORM'")
    #     e = sys.exc_info()
    #     print(e)
    #     errors.append(e)
    
    # try:
    #     ex_one(noise_test_path, noise_train_path, folder_extension="experiment_two_eval_noise", model_paths=noise_paths, data_to_test_on=data_to_test_on)
    #     ex_two_eval_noise(test_path, 'experiment_two_eval_noise', model_paths=noise_paths, data_to_test_on=data_to_test_on)
    # except:
    #     print("ERROR IN EXPERIMENT 'TRAIN ON NOISE'")
    #     e = sys.exc_info()
    #     print(e)
    #     errors.append(e)

    sum_merged_files('phase_two/csv_output')


    if len(errors) != 0:
        time_str = time.strftime("%Y%m%d-%H%M%S")
        with open(f"error_messages/output_error_{time_str}.txt", 'w') as output:
            output.write(str(errors))
