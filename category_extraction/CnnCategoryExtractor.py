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

param_grid = dict(params)

"""
param_grid = {
    'epochs': [50],
    'batch_size': [64],
    'validation_split': [0.15],
    'filters': [320],
    'kernel_size': [5],
    'conv_activation': ['relu'],
    'conv_l2_regularizer': [0.001],
    'dropout_rate': [0.6, 0.8],
    'dense_activation': ['tanh'],
    'dense_l2_regularizer': [0.001],
    'activation': ['sigmoid'],
    'optimizer': ['nadam'],
    'loss_function': ['binary_crossentropy'],
    'units': [256]
}
"""

"""
param_grid = {
        'epochs': [1],
        'batch_size': [64],
        'filters': [64],
        'kernel_size': [3],
        'conv_activation': ['relu'],
        'conv_l2_regularizer': [0.01],
        'dropout_rate': [0.6],
        'dense_activation': ['relu'],
        'dense_l2_regularizer': [0.01],
        'activation': ['sigmoid'],
        'optimizer': ['nadam'],
        'loss_function': ['binary_crossentropy'],
        'units': [256]
    }
"""

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
        self.cnn_model.save(self.MODEL_PATH)
        # self.cnn_model.summary()
        mode = kwargs.get('mode', 'train_validate_split')
        if mode == "train_validate_split":
            self.cnn_model.fit(
                X, y,
                **kwargs
            )
    
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
        layer_embedding = self._load_embedding(self.WE_PATH, trainable=True, vocabulary_size=15000, embedding_vector_length=500)(layer_input)
        layer_conv = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation=conv_activation,
        kernel_regularizer=regularizers.l2(conv_l2_regularizer))(layer_embedding)
        layer_pooling = GlobalMaxPooling1D()(layer_conv)
        layer_dropout_1 = Dropout(dropout_rate, seed=7)(layer_pooling)
        layer_dense_1 = Dense(units, activation=dense_activation, kernel_regularizer=regularizers.l2(dense_l2_regularizer))(layer_dropout_1)
        layer_dropout_2 = Dropout(dropout_rate, seed=7)(layer_dense_1)
        layer_softmax = Dense(4, activation=activation)(layer_dropout_2)
        
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
    
    def _plot_confusion_matrix(self,
                          cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
    def plot_all_confusion_matrix(self, y_test, y_pred):
        plt.figure()
        self._plot_confusion_matrix(confusion_matrix(y_test[:,0], y_pred[:,0]), classes=['0', '1'], title="O")
        plt.figure()
        self._plot_confusion_matrix(confusion_matrix(y_test[:,1], y_pred[:,1]), classes=['0', '1'], title="ASPECT-B")
        plt.figure()
        self._plot_confusion_matrix(confusion_matrix(y_test[:,2], y_pred[:,2]), classes=['0', '1'], title="ASPECT-I")
        plt.show()


def cnn():
    """
        Initialize data
    """
    X, y, X_test, y_test = utils.get_ce_dataset()

    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=7)

    """
        Make the model
    """
    np.random.seed(7)

    # checkpointer = ModelCheckpoint(filepath='model/cnn/weights/CNN.hdf5', verbose=1, save_best_only=True)
    ce = CNNCategoryExtractor()

    """
        Fit the model
    """
    # ce._fit_cv(X, y)
    # ce._fit_gridsearch_cv(X, y, param_grid, is_refit='f1_macro')
    ce.fit(X, y, verbose=1,
        epochs = 50,
        batch_size = 64,
        # validation_split = 0.15,
        filters = 320,
        kernel_size = 5,
        conv_activation = 'relu',
        conv_l2_regularizer = 0.001,
        dropout_rate = 0.5,
        dense_activation = 'tanh',
        dense_l2_regularizer = 0.001,
        activation = 'sigmoid',
        optimizer = "nadam",
        loss_function = 'binary_crossentropy',
        units = 256,
    )

    """
        Load best estimator and score it
    """
    # ce._fit_new_gridsearch_cv(X, y, params)
    # best_model = load_model('model/cnn/best.model')
    # del ce.cnn_model
    # ce.cnn_model = best_model
    ce.score(X_test, y_test, verbose=1)
    ce.threshold = 0.5
    ce.score(X_test, y_test, verbose=1)
    ce.threshold = 0.6
    ce.score(X_test, y_test, verbose=1)
    ce.threshold = 0.7
    ce.score(X_test, y_test, verbose=1)
    ce.threshold = 0.8
    ce.score(X_test, y_test, verbose=1)
    ce.threshold = 0.85
    ce.score(X_test, y_test, verbose=1)


if __name__ == "__main__":
    import time
    start_time = time.time()
    cnn()
    print("DONE IN {} seconds".format(time.time() - start_time))
