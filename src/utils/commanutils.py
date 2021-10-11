import tensorflow as tf
from tensorflow import keras as kr
import yaml
import time 
import os 


def get_config(config_path):
    with open(config_path) as config_file:
        content=yaml.safe_load(config_file)

    return content

def unique_filename(filename):
    unique_filename=time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename


def model_save(model,model_name,model_dir):
    unique_file=unique_filename(model_name)
    file_path=os.path.join(model_dir,unique_file)
    model.save(file_path)









    


