import time

from slave_main import run_biksture
from final_avg import calc_avg_from_experiments
from global_paths import get_paths

def get_phase_one_csv():
    return ['fitdata_combined.csv', 'test_val_sum_class_accuracy.csv']

def get_phase_two_csv():
    return ['final_sum_sum_sum_summed.csv' , 'sum_cat', 'sum_sub_cat.csv']

if __name__ == "__main__":
    s_index = 0
    e_index = 10
    data_to_test_on = 1
    
    phase_one_path = get_paths('phase_one_csv')
    phase_two_path = get_paths('phase_two_csv')
    
    phase_one_csv = get_phase_one_csv()
    phase_two_csv = get_phase_two_csv()
    
    output_folder = get_paths('result_csv')
    
    tic = time.time()
    
    for i in range(s_index, e_index):
        run_biksture(i, data_to_test_on, 
                    run_base_experiments=False,
                    run_ideal_experiments=True, 
                    run_lobster_experiments=True, 
                    run_lobster_level_experiments=True,
        )
    
    toc = time.time()
    
    tic_toc = toc - tic
    
    try:
        calc_avg_from_experiments(phase_one_path, phase_one_csv, phase_two_path, phase_two_csv, output_folder)
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("DONE! ")
    print(f"Time: {tic_toc / 60 / 60}")