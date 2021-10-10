import tensorflow as tf 
from tensorflow import keras as kr
from src.utils.commanutils import get_config

def model_creation(length_hiddenlayer1,length_hiddenlayer2,optimizer,loss,metrics):
    layers = [
            kr.layers.Flatten(input_shape=[28,28],name="INPUTLAYER"),
            kr.layers.Dense(length_hiddenlayer1,activation='relu',name="HIDDENLAYER1"),
            kr.layers.Dense(length_hiddenlayer2,activation='relu',name="HIDDENLAYER2"),
            kr.layers.Dense(10,activation='softmax',name="FINALLAYER")


    ]
    model_clf = kr.models.Sequential(layers)
    model_clf.summary()
    model_clf.compile(optimizer=optimizer,loss=loss,metrics=metrics)

    return model_clf

