params = [
    ('epochs', [75]),
    ('batch_size', [64]),
    ('validation_split', [0.]),
    ('dropout_rate', [0., 0.2, 0.5, 0.8]),
    ('dense_activation', ['relu']),
    ('dense_l2_regularizer', [0.01]),
    ('activation', ['softmax']),
    ('optimizer', ["nadam"]),
    ('loss_function', ['categorical_crossentropy']),
    ('gru_units', [64, 256]),
    ('units', [64, 256]),
    ('trainable', [False]),
    ('dense_layers', [1, 2, 3])
]

"""
params = [
    ('epochs', [1]),
    ('batch_size', [64]),
    ('recurrent_dropout', [0.9]),
    ('dropout_rate', [0.9]),
    ('dense_activation', ['relu']),
    ('dense_l2_regularizer', [0.01]),
    ('activation', ['sigmoid']),
    ('optimizer', ["nadam"]),
    ('loss_function', ['binary_crossentropy']),
    ('gru_units', [1]),
    ('units', [1]),
]
"""

param_grid = dict(params)

import itertools
import os
import sys

try:
    from constants import Const
    sys.path.insert(0, Const.ROOT)
except:
    sys.path.insert(0, '..')
    from constants import Const


import dill
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras import optimizers, regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers import (GRU, LSTM, RNN, Bidirectional, CuDNNGRU, CuDNNLSTM,
                          Dense, Dropout, Lambda, RepeatVector,
                          TimeDistributed, Concatenate)
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

from MyClassifier import KerasClassifier, MultilabelKerasClassifier, MyClassifier, Model
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


class RNNOpinionTargetExtractor (MyClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.MODEL_PATH = Const.OTE_ROOT + 'model/brnn/BRNN.model'
        self.WE_PATH = Const.WE_ROOT + 'embedding_matrix.pkl'
       
        self.target_names = ['O', 'B-ASPECT', 'I-ASPECT']
        self.rnn_model = None
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def fit(self, X, y,
        dropout_rate = 0.6,
        dense_activation = 'tanh',
        dense_l2_regularizer = 0.01,
        activation = 'sigmoid',
        optimizer = "nadam",
        loss_function = 'binary_crossentropy',
        gru_units = 256,
        units = 256,
        is_save = False,
        trainable = False,
        dense_layers = 1,
        **kwargs):

        self.rnn_model = self._create_model(
            dropout_rate = dropout_rate,
            dense_activation = dense_activation,
            dense_l2_regularizer = dense_l2_regularizer,
            activation = activation,
            optimizer = optimizer,
            loss_function = loss_function,
            gru_units = gru_units,
            units = units,
            trainable = trainable,
            dense_layers = dense_layers,
        )
        mode = kwargs.get('mode', 'train_validate_split')
        if mode == "train_validate_split":
            self.rnn_model.fit(
                X, y,
                **kwargs
            )
        
        if is_save:
            self.rnn_model.save(self.MODEL_PATH)
    
    def predict(self, X, **kwargs):
        y_pred = self.rnn_model.predict(X)
        return y_pred
    
    def score(self, X, pos, y, verbose=1, dense_layers=1, **kwargs):
        if self.rnn_model != None:
            rnn_model = self.rnn_model
        else:
            print("Scoring using untrained model")
            rnn_model = self._create_model()
        y_test = y

        def max_index(cat):
            i_max = -1
            val_max = -1
            for i, y in enumerate(cat):
                if val_max < y:
                    i_max = i
                    val_max = y
            return i_max

        def get_decreased_dimension(y, end):
            tmp = []
            for i, y_sent in enumerate(y):
                for y_token in y_sent[:end[i]]:
                    tmp.append(y_token)
            tmp = np.array(tmp)
            return tmp

        y_pred_raw = rnn_model.predict([X, pos], batch_size=1, verbose=verbose)
        y_pred = []

        for y_pred_raw_sents in y_pred_raw:
            y_pred_sents = []
            for y_pred_raw_tokens in y_pred_raw_sents:
                max_i = max_index(y_pred_raw_tokens)
                y = [0.] * len(self.target_names) #number of classes to be predicted
                y[max_i] = 1.
                y_pred_sents.append(y)
            y_pred.append(y_pred_sents)
        y_pred = np.array(y_pred)
        # y_pred = np.argmax(get_decreased_dimension(y_pred_raw), axis=-1)
        # y_test = np.argmax(get_decreased_dimension(y_test), axis=-1)
        
        end = utils.get_sentence_end_index(X)
        y_pred = get_decreased_dimension(y_pred, end)
        y_test = get_decreased_dimension(y_test, end)

        f1_score_macro = f1_score(y_test, y_pred, average='macro')
        precision_score_macro = precision_score(y_test, y_pred, average='macro')
        recall_score_macro = recall_score(y_test, y_pred, average='macro')
        f1_scores = f1_score(y_test, y_pred, average=None)
        precision_scores = precision_score(y_test, y_pred, average=None)
        recall_scores = recall_score(y_test, y_pred, average=None)
        accuracy = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

        scores = {
            'f1_score_macro': f1_score_macro,
            'precision_score_macro': precision_score_macro,
            'recall_score_macro': recall_score_macro,
            'f1_scores': f1_scores,
            'precision_scores': precision_scores,
            'recall_scores': recall_scores,
            'accuracy': accuracy,
            'confusion_matrix': conf_mat
        }

        if verbose > 0:
            print("F1-Score  : {}".format(f1_scores))
            print("Precision : {}".format(precision_scores))
            print("Recall    : {}".format(recall_scores))
            print("Accuracy  : {}".format(accuracy))
            print("F1-Score-Macro:", f1_score_macro)
            print("P -Score-Macro:", precision_score_macro)
            print("R -Score-Macro:", recall_score_macro)
            print("Confusion Matrix:\n", conf_mat)
            print()
        return scores

    def _fit_train_validate_split(self, X, y):
        pass

    def _create_model(
        self,
        dropout_rate = 0.6,
        dense_activation = 'tanh',
        dense_l2_regularizer = 0.01,
        activation = 'sigmoid',
        optimizer = "nadam",
        loss_function = 'binary_crossentropy',
        gru_units = 256,
        units = 256,
        trainable = False,
        dense_layers = 1,

        **kwargs
    ):
        K.clear_session()
        MAX_SEQUENCE_LENGTH = 81
        # Define Architecture
        layer_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        layer_embedding = self._load_embedding(self.WE_PATH, trainable=trainable, vocabulary_size=15000, embedding_vector_length=500)(layer_input)
        layer_input_pos = Input(shape=(MAX_SEQUENCE_LENGTH,18))
        layer_concat = Concatenate()([layer_embedding, layer_input_pos])
        layer_blstm = Bidirectional(LSTM(gru_units, return_sequences=True, recurrent_dropout=dropout_rate, stateful=False))(layer_concat)
        layer_dropout = TimeDistributed(Dropout(dropout_rate, seed=7))(layer_blstm)
        for i in range(dense_layers):
            layer_dense = TimeDistributed(Dense(units, activation=dense_activation, kernel_regularizer=regularizers.l2(dense_l2_regularizer)))(layer_dropout)
            layer_dropout = TimeDistributed(Dropout(dropout_rate, seed=7))(layer_dense)
            
        layer_softmax = TimeDistributed(Dense(3, activation=activation))(layer_dropout)
        rnn_model = Model(inputs=[layer_input, layer_input_pos], outputs=layer_softmax)

        # Create Optimizer
        # optimizer = optimizers.SGD(lr=0.05, momentum=0.9, decay=0.0, nesterov=True)
        rnn_model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'],
                        sample_weight_mode="temporal")

        rnn_model.summary()
        return rnn_model
        
        # layer_input = Input(shape=(max_review_length,))
        # layer_embedding = kwargs.get('layer_embedding', Embedding(n_words, embedding_vector_length))(layer_input)
        # layer_lstm = LSTM(256, recurrent_dropout=0.2)(layer_embedding)
        # layer_repeat = RepeatVector(max_review_length)(layer_lstm)
        # layer_blstm = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.0))(layer_repeat)
        # layer_dropout_1 = TimeDistributed(Dropout(0.6, seed=7))(layer_blstm)
        # layer_dense_1 = TimeDistributed(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))(layer_dropout_1)
        # layer_softmax = TimeDistributed(Dense(3, activation='softmax'))(layer_dense_1)
        # rnn_model = Model(inputs=layer_input, outputs=layer_softmax)
    
    def _get_features(self, x):
        return x

    def load_weights(self, path):
        self._create_model()
        self.rnn_model.load_weights(path)

    def load_best_model(self):
        best_model = load_model(Const.OTE_ROOT + 'model/brnn/best.model')
        del self.rnn_model
        self.rnn_model = best_model
        

def get_params_from_grid(param_grid):
    import itertools as it
    all_names = sorted(param_grid)
    combinations = it.product(*(param_grid[Name] for Name in all_names))
    return combinations

def main():
    import numpy as np

    """
        Initialize data
    """
    X, y, pos, X_test, y_test, pos_test = utils.get_ote_dataset()
    # X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.15, random_state=7)
    
    """
        Calculate Sample Weight
    """
    sample_weight = utils.get_sample_weight(X, y, mu=0.1, threshold=1.)
    print(sample_weight)
    
    """
        Make and fit the model
    """
    np.random.seed(7)

    ote = RNNOpinionTargetExtractor()

    # n_epoch = 25
    # for i in range(n_epoch):
    #     print('Epoch #', i)
    #     model.fit(x=X_train, y=y_train, epochs=1, batch_size=32,
    #           validation_data=(X_validate, y_validate), callbacks=[checkpointer]
    #           ,sample_weight=sample_weight)
    #     model.reset_states()

    # params_for_fit = {
    #     "dropout_rate": 0.5,
    #     "dense_activation": 'relu',
    #     "dense_l2_regularizer": 0.01,
    #     "activation": 'softmax',
    #     "optimizer": 'nadam',
    #     "loss_function": 'categorical_crossentropy',
    #     "gru_units": 64,
    #     "units": 64,
    #     'dense_layers' : 1,
    # }
    # ote.fit(
    #     [X, pos], y,
    #     epochs = 100,
    #     batch_size = 64,
    #     **params_for_fit,
    #     sample_weight = sample_weight,
    #     is_save=True,
    # )
    # ote.score(X_test, pos_test, y_test, dense_layers = 1)

    # ote._fit_new_gridsearch_cv(X, y, params, sample_weight=sample_weight, score_verbose=1)

    """
        Load best estimator and score it
    """
    ote.load_best_model()
    ote.score(X_test, pos_test, y_test, dense_layers = 1)
    
if __name__ == "__main__":
    utils.time_log(main)