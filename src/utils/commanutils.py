import tensorflow as tf
from tensorflow import keras as kr
import yaml


def get_config(config_path):
    with open(config_path) as config_file:
        content=yaml.safe_load(config_file)

    return content
    


    


