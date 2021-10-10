import tensorflow as tf 
from tensorflow import keras as kr
from src.utils.commanutils import get_config


def get_data(Validation_data_length):
    mnist=kr.datasets.mnist
    (X_train_full,y_train_full),(X_test,y_test)=mnist.load_data()
    X_valid=X_train_full[:Validation_data_length]/255
    y_valid=y_train_full[:Validation_data_length]
    X_train=X_train_full[Validation_data_length:]/255
    y_train=y_train_full[Validation_data_length:]
    
    X_test=X_test/255

    return (X_train,y_train),(X_valid,y_valid),(X_test,y_test)