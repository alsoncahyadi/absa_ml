import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
sys.path.insert(0, '..')

import utils

from MyClassifier import MyClassifier, MultilabelKerasClassifier
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

class CNNCategoryExtractor (MyClassifier):
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
        # self.cnn_model.summary()
        mode = kwargs.get('mode', 'train_validate_split')
        if mode == "train_validate_split":
            self.cnn_model.fit(
                X, y,
                **kwargs
            )
            self.cnn_model.load_weights(self.WEIGHTS_PATH)
    
    def predict(self, X, **kwargs):
        y_pred = self.cnn_model.predict(X)
        THRESHOLD = kwargs.get('threshold', 0.75)
        y_pred[y_pred >= THRESHOLD] = 1.
        y_pred[y_pred < THRESHOLD] = 0.
        return y_pred
    
    def score(self, X, y, **kwargs):
        # del self.cnn_model
        # self.cnn_model = load_model(self.MODEL_PATH)
        # self.cnn_model.load_weights(self.WEIGHTS_PATH)
        # Final evaluation of the model
        scores = self.cnn_model.evaluate(X, y, verbose=0)
        print("Test Set Accuracy: %.2f%%" % (scores[1]*100))

        y_pred = self.predict(X)

        AVERAGE = None
        print("F1-Score  : {}".format(f1_score(y, y_pred, average=AVERAGE)))
        print("Precision : {}".format(precision_score(y, y_pred, average=AVERAGE)))
        print("Recall    : {}".format(recall_score(y, y_pred, average=AVERAGE)))
        print("Accuracy  : {}".format(accuracy_score(y, y_pred)))

        f1_score_macro = f1_score(y, y_pred, average='macro')
        print("F1-Score-Macro:", f1_score_macro)

        is_show_confusion_matrix = kwargs.get('show_confusion_matrix', False)
        if is_show_confusion_matrix:
            self.plot_all_confusion_matrix(y, y_pred)
        
        return f1_score_macro

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

        **kwargs
    ):
        K.clear_session()
        MAX_SEQUENCE_LENGTH = kwargs.get("max_sequence_length")
        print(MAX_SEQUENCE_LENGTH)

        # Define Architecture
        layer_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        # layer_feature = Lambda(self._get_features)(layer_input)
        layer_embedding = self.layer_embedding(layer_input)
        layer_conv = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation=conv_activation,
        kernel_regularizer=regularizers.l2(conv_l2_regularizer))(layer_embedding)
        layer_pooling = GlobalMaxPooling1D()(layer_conv)
        layer_dropout_1 = Dropout(dropout_rate, seed=7)(layer_pooling)
        layer_dense_1 = Dense(256, activation=dense_activation, kernel_regularizer=regularizers.l2(dense_l2_regularizer))(layer_dropout_1)
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


class CategoryFeatureExtractor (BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        raise NotImplementedError

class BinCategoryExtractor (MyClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.WEIGHTS_PATH = 'model/cnn/weights/CNN.hdf5'
        self.MODEL_PATH = 'model/cnn/CNN.model'
        self.WE_PATH = '../we/embedding_matrix.pkl'
       
        self.layer_embedding = self._load_embedding(self.WE_PATH, trainable=True, vocabulary_size=15000, embedding_vector_length=500)
        self.model = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _create_ann_model(
        self,
        dropout_rate = 0.6,
        dense_activation = 'relu',
        dense_l2_regularizer = 0.01,
        activation = 'sigmoid',
        optimizer = "nadam",
        loss_function = 'binary_crossentropy',

        **kwargs
    ):
        MAX_SEQUENCE_LENGTH = kwargs.get("max_sequence_length", 150)
        # Define Architecture
        layer_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        layer_dropout_1 = Dropout(dropout_rate, seed=7)(layer_input)
        layer_dense_1 = Dense(256, activation=dense_activation, kernel_regularizer=regularizers.l2(dense_l2_regularizer))(layer_dropout_1)
        layer_softmax = Dense(1, activation=activation)(layer_dense_1)
        
        # Create Model
        ann_model = Model(inputs=layer_input, outputs=layer_softmax)
        
        # Create Optimizer
        ann_model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
        ann_model.summary()
        return ann_model

    
    def fit(self, X, y):
        from sklearn.multiclass import OneVsRestClassifier
        from keras.wrappers.scikit_learn import MultilabelKerasClassifier
        ann_model_sk = MultilabelKerasClassifier(build_fn = self._create_ann_model, verbose=0)
        ovr = OneVsRestClassifier(ann_model_sk)
        ovr.fit(X, y)
        self.model = ovr
    
    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y, **kwargs):
        y_pred = self.predict(X)

        AVERAGE = None
        print("F1-Score  : {}".format(f1_score(y, y_pred, average=AVERAGE)))
        print("Precision : {}".format(precision_score(y, y_pred, average=AVERAGE)))
        print("Recall    : {}".format(recall_score(y, y_pred, average=AVERAGE)))
        print("Accuracy  : {}".format(accuracy_score(y, y_pred)))

        f1_score_macro = f1_score(y, y_pred, average='macro')
        print("F1-Score-Macro:", f1_score_macro)

        # is_show_confusion_matrix = kwargs.get('show_confusion_matrix', False)
        # if is_show_confusion_matrix:
        #     self.plot_all_confusion_matrix(y, y_pred)
        
        return f1_score_macro


def binDriver():
    """
        Initialize data
    """
    X, y, X_test, y_test = utils.get_ce_dataset()
    
    """
        Make the model
    """
    np.random.seed(7)
    bin = BinCategoryExtractor()
    print(X.shape, y.shape)
    bin.fit(X, y)
    bin.score(X_test, y_test)

def main():
    """
        Initialize data
    """
    X, y, X_test, y_test = utils.get_ce_dataset()

    # X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=7)

    """
        Make the model
    """
    np.random.seed(7)

    # checkpointer = ModelCheckpoint(filepath='model/cnn/weights/CNN.hdf5', verbose=1, save_best_only=True)
    ce = CNNCategoryExtractor()

    """
        Fit the model
    """
    param_grid = {
        'epochs': [50],
        'batch_size': [64],
        'filters': [320],
        'kernel_size': [5],
        'conv_activation': ['relu', 'tanh'],
        'conv_l2_regularizer': [0.01],
        'dropout_rate': [0.6],
        'dense_activation': ['relu', 'tanh'],
        'dense_l2_regularizer': [0.01],
        'activation': ['sigmoid'],
        'optimizer': ['nadam'],
        'loss_function': ['binary_crossentropy']
    }
    ce._fit_gridsearch_cv(X, y, param_grid, is_refit='f1_macro')

    best_model = None
    with open('model/cnn/best.model', 'rb') as fi:
        best_model = dill.load(fi)
    del ce.cnn_model
    ce.cnn_model = best_model
    ce.score(X_test, y_test)

if __name__ == "__main__":
    # binDriver()
    main()