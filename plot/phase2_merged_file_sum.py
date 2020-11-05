from generalized_sum import generalized_sum
from write_csv_file import cvs_object
from sum_constructors.phase2_sum_constructor import get_object as sum_con
from sum_constructors.phase_2_sum_summed_constructor import get_object as sum_summed_con

def split_path(csv_path):
    return_val = ''
    split_path = csv_path.split('/')[:-1]
    
    for i in range(len(split_path)):
        return_val += str(split_path[i]) + '/'
    
    return return_val

def sum_merged_file(get_category, get_sub_category, get_class_accuracy):
    csv_path = 'phase_two/csv_output/phase2_merged_file.csv' 
    
    sum_path = f'{split_path(csv_path)}/phase2_sum_sub_cat.csv'
    sum_summed_path = f'{split_path(csv_path)}/phase2_sum_cat.csv'
    
    csv_obj = cvs_object(csv_path)
    data = generalized_sum(csv_obj, sum_con(get_sub_category, get_category, get_class_accuracy))
    csv_obj.write(data, path=sum_path, overwrite_path=True)
    data = generalized_sum(csv_obj, sum_summed_con(get_sub_category, get_category, get_class_accuracy))
    csv_obj.write(data, path=sum_summed_path, overwrite_path=True)
    
