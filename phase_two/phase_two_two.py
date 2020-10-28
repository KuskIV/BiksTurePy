import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from phase_one.find_ideal_model import train_and_eval_models_for_size
from global_paths import get_paths
from Noise_Generators.noise_main import Filter,premade_single_filter,apply_multiple_filters
from Dataset.load_h5 import h5_object

def train_noise_model():
    pass
def generate_noise_dataset(h5_path,dataset_split, lazy_split=1,lazy_start=0):
    h5_obj = h5_object(h5_path, training_split=dataset_split)
    original_images, original_labels, _, _ = h5_obj.shuffle_and_lazyload(lazy_start, lazy_split)
    
def test_model(model,test_path):
    pass
def create_csv():
    pass