import itertools
import os
import sys

sys.path.insert(0, '..')


import dill
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras import optimizers, regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers import (GRU, LSTM, RNN, Bidirectional, CuDNNGRU, CuDNNLSTM,
                          Dense, Dropout, Lambda, RepeatVector,
                          TimeDistributed)
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.layers.pooling import (AveragePooling1D, GlobalMaxPooling1D,
                                  MaxPooling1D)
from keras.models import Input, Sequential, load_model
from keras.preprocessing import sequence, text
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

import utils
from MyClassifier import KerasClassifier, MultilabelKerasClassifier, MyClassifier, Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

sys.path.insert(0, '..')


class CRFOpinionTargetExtractor ():
    def __init__(self):
        super(CRFOpinionTargetExtractor).__init__()
    
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def _create_model(self):
        pass

def crf():
    pass

if __name__ == "__main__":
    utils.time_log(crf)