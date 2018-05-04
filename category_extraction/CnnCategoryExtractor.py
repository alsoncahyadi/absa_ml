params = [
    ('epochs', [50]),
    ('batch_size', [64]),
    ('validation_split', [0.15]),
    ('filters', [320, 64]),
    ('kernel_size', [5, 3]),
    ('conv_activation', ['relu', 'tanh']),
    ('conv_l2_regularizer', [0.01, 0.001]),
    ('dropout_rate', [0.6, 0.8]),
    ('dense_activation', ['relu', 'tanh']),
    ('dense_l2_regularizer', [0.01, 0.001]),
    ('activation', ['sigmoid']),
    ('optimizer', ['nadam']),
    ('loss_function', ['binary_crossentropy']),
    ('units', [256, 64, 16]),
]

"""
params = [
    ('epochs', [1]),
    ('batch_size', [128]),
    ('validation_split', [0.]),
    ('filters', [1]),
    ('kernel_size', [3]),
    ('conv_activation', ['relu']),
    ('conv_l2_regularizer', [0.]),
    ('dropout_rate', [0.9]),
    ('dense_activation', ['relu']),
    ('dense_l2_regularizer', [0.]),
    ('activation', ['sigmoid']),
    ('optimizer', ['nadam']),
    ('loss_function', ['binary_crossentropy']),
    ('units', [4]),
]
"""
param_grid = dict(params)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
sys.path.insert(0, '..')

import utils
from ItemSelector import ItemSelector

from MyClassifier import MyClassifier, MultilabelKerasClassifier, KerasClassifier, Model
from MyOneVsRestClassifier import MyOneVsRestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from keras import backend as K
from keras.models import Sequential, Input, load_model
from keras.layers import Dense, LSTM, Flatten, Dropout, Lambda, BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence, text
from keras import regularizers, optimizers
from keras.callbacks import ModelCheckpoint

import dill
import numpy as np

import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

class CNNCategoryExtractor (MyClassifier):
    def __init__(self, threshold = 0.75, **kwargs):
        super().__init__(**kwargs)

        self.WEIGHTS_PATH = 'model/cnn/weights/CNN.hdf5'
        self.MODEL_PATH = 'model/cnn/CNN.model'
        self.WE_PATH = '../we/embedding_matrix.pkl'
       
        self.target_names = ['food', 'service', 'price', 'place']
        self.cnn_model = None
        self.threshold = threshold
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def fit(self, X, y,
        
        filters = 320,
        kernel_size = 5,
        conv_activation = 'tanh',
        conv_l2_regularizer = 0.01,
        dropout_rate = 0.6,
        dense_activation = 'relu',
        dense_l2_regularizer = 0.01,
        activation = 'sigmoid',
        optimizer = "nadam",
        loss_function = 'binary_crossentropy',
        units = 256,

        is_save = False,
        
        **kwargs):

        self.cnn_model = self._create_model(
            filters,
            kernel_size,
            conv_activation,
            conv_l2_regularizer,
            dropout_rate,
            dense_activation,
            dense_l2_regularizer,
            activation,
            optimizer,
            loss_function,
            units,
        )
        # self.cnn_model.save(self.MODEL_PATH)
        # self.cnn_model.summary()
        mode = kwargs.get('mode', 'train_validate_split')
        if mode == "train_validate_split":
            self.cnn_model.fit(
                X, y,
                **kwargs
            )
        if is_save:
            self.cnn_model.save(self.MODEL_PATH)
    
    def predict(self, X):
        threshold = self.threshold
        y_pred = self.cnn_model.predict(X)
        y_pred[y_pred >= threshold] = 1.
        y_pred[y_pred < threshold] = 0.
        return y_pred

    def _fit_train_validate_split(self, X, y):
        pass

    def _fit_gridsearch_cv(self, X, y, param_grid, **kwargs):
        from sklearn.model_selection import GridSearchCV
        np.random.seed(7)
        # Wrap in sklearn wrapper
        model = MultilabelKerasClassifier(build_fn = self._create_model, verbose=0)
        # model.fit(X, y)
        # print(model.predict(X))

        # train
        IS_REFIT = kwargs.get('is_refit', 'f1_macro')
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, refit=IS_REFIT, verbose=1, scoring=['f1_macro', 'precision_macro', 'recall_macro'])
        grid_result = grid.fit(X, y)
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        print(grid_result.cv_results_.keys())
        means = [grid_result.cv_results_['mean_test_f1_macro'], grid_result.cv_results_['mean_test_precision_macro'], grid_result.cv_results_['mean_test_recall_macro']]
        stds = [grid_result.cv_results_['std_test_f1_macro'], grid_result.cv_results_['std_test_precision_macro'], grid_result.cv_results_['std_test_recall_macro']]
        for mean, stdev in zip(means, stds):
            print("\n{} ({})".format(mean, stdev))
        params = grid_result.best_params_
        print("with:", params)
        with open('output/gridsearch_cnn.pkl', 'wb') as fo:
            dill.dump(grid_result.cv_results_, fo)
        if IS_REFIT:
            grid.best_estimator_.model.save('model/cnn/best.model')

    def _create_model(
        self,

        filters = 320,
        kernel_size = 5,
        conv_activation = 'tanh',
        conv_l2_regularizer = 0.01,
        dropout_rate = 0.6,
        dense_activation = 'relu',
        dense_l2_regularizer = 0.01,
        activation = 'sigmoid',
        optimizer = "nadam",
        loss_function = 'binary_crossentropy',
        units = 256,

        **kwargs
    ):
        K.clear_session()
        MAX_SEQUENCE_LENGTH = kwargs.get("max_sequence_length", 150)

        # Define Architecture
        layer_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        layer_embedding = self._load_embedding(self.WE_PATH, trainable=False, vocabulary_size=15000, embedding_vector_length=500)(layer_input)
        layer_conv = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation=conv_activation,
        kernel_regularizer=regularizers.l2(conv_l2_regularizer))(layer_embedding)
        layer_pooling = GlobalMaxPooling1D()(layer_conv)
        layer_dropout_1 = Dropout(dropout_rate, seed=7)(layer_pooling)
        layer_dense_1 = Dense(units, activation=dense_activation, kernel_regularizer=regularizers.l2(dense_l2_regularizer))(layer_dropout_1)
        # layer_dropout_2 = Dropout(dropout_rate, seed=7)(layer_dense_1)
        layer_softmax = Dense(4, activation=activation)(layer_dense_1)
        
        # Create Model
        cnn_model = Model(inputs=layer_input, outputs=layer_softmax)
        
        # Create Optimizer
        # optimizer = optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        # optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        cnn_model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
        return cnn_model
    
    def _get_features(self, x):
        return x

    def load_weights(self, path):
        self._create_model()
        self.cnn_model.load_weights(path)

    def set_threshold(self, thresh):
        self.threshold = thresh

    def get_threshold(self):
        return self.threshold

def cnn():
    """
        Initialize data
    """
    X, y, X_test, y_test = utils.get_ce_dataset()

    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=7)
    thresh_to_try = [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.999]
    """
        Make the model
    """
    np.random.seed(7)

    # checkpointer = ModelCheckpoint(filepath='model/cnn/weights/CNN.hdf5', verbose=1, save_best_only=True)
    ce = CNNCategoryExtractor()

    """
        Fit the model
    """
    
    ce.fit(X_train, y_train, verbose=1,
        epochs = 50,
        batch_size = 64,
        # validation_split = 0.15,
        filters = 320,
        kernel_size = 5,
        conv_activation = 'relu',
        conv_l2_regularizer = 0.001,
        dropout_rate = 0.6,
        dense_activation = 'tanh',
        dense_l2_regularizer = 0.001,
        activation = 'sigmoid',
        optimizer = "nadam",
        loss_function = 'binary_crossentropy',
        units = 256,
        is_save = True
    )
    
    # ce._fit_new_gridsearch_cv(X, y, params, thresholds=thresh_to_try, score_verbose=True)

    """
        Load best estimator and score it
    """
    best_model = load_model(ce.MODEL_PATH)
    del ce.cnn_model
    ce.cnn_model = best_model
    for thresh in thresh_to_try:
        print("\nTHRESH: {}".format(thresh))
        ce.set_threshold(thresh); ce.score(X_test, y_test)


if __name__ == "__main__":
    utils.time_log(cnn)
