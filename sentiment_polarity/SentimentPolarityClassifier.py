# grid search hypers

param_grid = {
    'epochs': [25, 50],
    'batch_size': [64],
    'filters': [320, 64],
    'kernel_size': [5, 3],
    'conv_activation': ['relu', 'tanh'],
    'conv_l2_regularizer': [0.01, 0.001],
    'dropout_rate': [0.6],
    'dense_activation': ['relu', 'tanh'],
    'dense_l2_regularizer': [0.01, 0.001],
    'activation': ['softmax'],
    'optimizer': ['nadam'],
    'loss_function': ['categorical_crossentropy'],
    'units': [256, 64, 16]
}

"""
param_grid = {
    'epochs': [1],
    'batch_size': [64],
    'filters': [1],
    'kernel_size': [3],
    'conv_activation': ['relu'],
    'conv_l2_regularizer': [0.],
    'dropout_rate': [0.9],
    'dense_activation': ['relu'],
    'dense_l2_regularizer': [0.],
    'activation': ['sigmoid'],
    'optimizer': ['nadam'],
    'loss_function': ['binary_crossentropy'],
    'units': [4]
}
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
sys.path.insert(0, '..')

import utils

from MyClassifier import MyClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from keras import backend as K
from keras.models import Sequential, Input, Model, load_model
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

class CNNSentimentPolarityClassifier (MyClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.WEIGHTS_PATH = 'model/cnn/weights/CNN.hdf5'
        self.MODEL_PATH = 'model/cnn/CNN.model'
        self.WE_PATH = '../we/embedding_matrix.pkl'
       
        self.layer_embedding = self._load_embedding(self.WE_PATH, trainable=True, vocabulary_size=15000, embedding_vector_length=500)
        self.cnn_model = None
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def fit(self, X, y, **kwargs):
        self.cnn_model = self._create_model()
        self.cnn_model.save(self.MODEL_PATH)
        mode = kwargs.get('mode', 'train_validate_split')
        if mode == "train_validate_split":
            self.cnn_model.fit(
                X, y,
                n_class = len(y[0]),
                **kwargs
            )
            self.cnn_model.load_weights(self.WEIGHTS_PATH)
    
    def predict(self, X, **kwargs):
        y_pred = self.cnn_model.predict(X)
        if len(y_pred.shape) > 2:
            return y_pred.argmax(axis=-1)
        else:
            THRESHOLD = kwargs.get('threshold', 0.75)
            y_pred[y_pred >= THRESHOLD] = 1.
            y_pred[y_pred < THRESHOLD] = 0.
            return y_pred

    def _fit_train_validate_split(self, X, y):
        pass

    def _fit_gridsearch_cv(self, X, y, param_grid, category="NOTSPECIFIED", **kwargs):
        from sklearn.model_selection import GridSearchCV
        from keras.wrappers.scikit_learn import KerasClassifier

        np.random.seed(7)

        # Wrap in sklearn wrapper
        model = KerasClassifier(build_fn = self._create_model, verbose=0)

        # train
        IS_REFIT = kwargs.get('is_refit','f1_macro')
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, refit=IS_REFIT, scoring=['f1_macro', 'precision_macro', 'recall_macro'], verbose=1)
        grid_result = grid.fit(X, y)
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        print(grid_result.cv_results_.keys())
        means = [grid_result.cv_results_['mean_test_f1_macro'], grid_result.cv_results_['mean_test_precision_macro'], grid_result.cv_results_['mean_test_recall_macro']]
        stds = [grid_result.cv_results_['std_test_f1_macro'], grid_result.cv_results_['std_test_precision_macro'], grid_result.cv_results_['std_test_recall_macro']]
        for mean, stdev in zip(means, stds):
            print("\n{} ({})".format(mean, stdev))
        params = grid_result.best_params_
        print("with:", params)
        with open('output/gridsearch_cnn_{}.pkl'.format(category), 'wb') as fo:
            dill.dump(grid_result.cv_results_, fo)
        if IS_REFIT:
            grid.best_estimator_.model.save('model/cnn/best_{}.model'.format(category))

    def _create_model(
        self,

        filters = 320,
        kernel_size = 5,
        conv_activation = 'tanh',
        conv_l2_regularizer = 0.01,
        dropout_rate = 0.6,
        dense_activation = 'relu',
        dense_l2_regularizer = 0.01,
        activation = 'softmax',
        optimizer = "nadam",
        loss_function = 'categorical_crossentropy',
        units = 256,

        **kwargs
    ):
        K.clear_session()
        MAX_SEQUENCE_LENGTH = kwargs.get("max_sequence_length")
        n_class = kwargs.get('n_class', 1)

        # Define Architecture
        layer_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        # layer_feature = Lambda(self._get_features)(layer_input)
        layer_embedding = self.layer_embedding(layer_input)
        layer_conv = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation=conv_activation,
        kernel_regularizer=regularizers.l2(conv_l2_regularizer))(layer_embedding)
        layer_pooling = GlobalMaxPooling1D()(layer_conv)
        layer_dropout_1 = Dropout(dropout_rate, seed=7)(layer_pooling)
        layer_dense_1 = Dense(units, activation=dense_activation, kernel_regularizer=regularizers.l2(dense_l2_regularizer))(layer_dropout_1)
        layer_softmax = Dense(n_class, activation=activation)(layer_dense_1)
        
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
        self._plot_confusion_matrix(confusion_matrix(y_test[:,0], y_pred[:,0]), classes=['Negative', 'Positive', ''], title="Sentiment")
        plt.show()


class CategoryFeatureExtractor (BaseEstimator):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        raise NotImplementedError
    
    def transform(self):
        raise NotImplementedError

def main():
    categories = ['food', 'service', 'price', 'place']
    for category in categories:
        print("\n\n========= CHECKING CATEGORY:", category, "==========")
        """
            Initialize data
        """
        X, y, X_test, y_test = utils.get_spc_dataset(category)
        # X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=7)

        """
            Make the model
        """
        np.random.seed(7)

        # checkpointer = ModelCheckpoint(filepath='model/cnn/weights/CNN.hdf5', verbose=0, save_best_only=True)
        spc = CNNSentimentPolarityClassifier()

        """
            Fit the model
        """
        

        # spc._fit_gridsearch_cv(X, y, param_grid, category)
        
        """
            Load best estimator and score it
        """

        best_model = load_model('model/cnn/best_{}.model'.format(category))
        del spc.cnn_model
        spc.cnn_model = best_model
        from keras.utils.np_utils import to_categorical
        spc.score(X_test, y_test)

if __name__ == "__main__":
    main()