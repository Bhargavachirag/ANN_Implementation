import tensorflow as tf
from tensorflow import keras as kr
from src.utils.data_managment import get_data
from src.utils.commanutils import get_config
from src.utils.model import model_creation
import os
import argparse


def training(config_path):
    config=get_config(config_path)
    Validation_data_length=config['params']['Validation_data_length']
    (X_train,y_train),(X_valid,y_valid),(X_test,y_test)=get_data(Validation_data_length)
    length_hiddenlayer1=config['params']['length_hiddenlayer1']
    length_hiddenlayer2=config['params']['length_hiddenlayer2']
    optimizer=config['params']['optimizer']
    loss=config['params']['loss']
    metrics=config['params']['metrics']

    model=model_creation(length_hiddenlayer1, length_hiddenlayer2, optimizer, loss, metrics)
    Detail_losses=model.fit(X_train,y_train,epochs=config['params']['epochs'],validation_data=(X_valid,y_valid))

    return model

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    print(args)

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()

    print(parsed_args)

    training(config_path=parsed_args.config)


     