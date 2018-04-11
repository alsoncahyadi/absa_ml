import sys
sys.path.insert(0, '..')

from MyClassifier import MyClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

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
        self.cnn_model.summary()
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
    
    def score(self, X, y, **kwargs):
        # del self.cnn_model
        # self.cnn_model = load_model(self.MODEL_PATH)
        # self.cnn_model.load_weights(self.WEIGHTS_PATH)
        # Final evaluation of the model
        scores = self.cnn_model.evaluate(X, y, verbose=0)
        print("Test Set Accuracy: %.2f%%" % (scores[1]*100))

        y_pred = self.cnn_model.predict(X)
        THRESHOLD = kwargs.get('threshold', 0.75)
        print("Threshold:", THRESHOLD)
        y_pred[y_pred >= THRESHOLD] = 1.
        y_pred[y_pred < THRESHOLD] = 0.

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

        **kwargs
    ):
    
        MAX_SEQUENCE_LENGTH = kwargs.get("max_sequence_length")

        # Define Architecture
        layer_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        # layer_feature = Lambda(self._get_features)(layer_input)
        layer_embedding = self.layer_embedding(layer_input)
        layer_conv = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation=conv_activation,
        kernel_regularizer=regularizers.l2(conv_l2_regularizer))(layer_embedding)
        layer_pooling = GlobalMaxPooling1D()(layer_conv)
        layer_dropout_1 = Dropout(dropout_rate, seed=7)(layer_pooling)
        layer_dense_1 = Dense(256, activation=dense_activation, kernel_regularizer=regularizers.l2(dense_l2_regularizer))(layer_dropout_1)
        layer_softmax = Dense(3, activation=activation)(layer_dense_1)
        
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

    def fit(self, x, y=None):
        raise NotImplementedError
    
    def transform(self):
        raise NotImplementedError

def main():

    """
        Load Tokenizer
    """
    # Make Tokenizer (load or from dataset)
    from keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer()

    with open('../we/tokenizer.pkl', 'rb') as fi:
        tokenizer = dill.load(fi)

    """
        Initialize data
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv("data/train_data.csv", delimiter=";", header=0, encoding = "ISO-8859-1")
    df_test = pd.read_csv("data/test_data.csv", delimiter=";", header=0, encoding = "ISO-8859-1")

    df = df.sample(frac=1, random_state=7)

    categories = ['food', 'service', 'price', 'place']

    X = df[df[categories[0]] != '-' ]['review']
    X_test = df_test[df_test[categories[0]] != '-' ]['review']

    X = tokenizer.texts_to_sequences(X)
    X_test = tokenizer.texts_to_sequences(X_test)

    max_review_length = 150
    PADDING_TYPE = 'post'
    X = sequence.pad_sequences(X, maxlen=max_review_length, padding=PADDING_TYPE)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length, padding=PADDING_TYPE)

    from keras.utils import to_categorical
    y = df[categories[0]]
    y = y[y != '-']
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(y)

    y_test = df_test[categories[0]]
    y_test = y_test[y_test != '-']
    y_test = le.transform(y_test)

    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=7)
    print(np.isnan(y_test).any())

    """
        Make the model
    """
    np.random.seed(7)

    checkpointer = ModelCheckpoint(filepath='model/cnn/weights/CNN.hdf5', verbose=0, save_best_only=True)
    spc = CNNSentimentPolarityClassifier()

    """
        Fit the model
    """
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from keras.wrappers.scikit_learn import KerasClassifier
    IS_FIT = True

    np.random.seed(7)

    # Wrap in sklearn wrapper
    model = KerasClassifier(build_fn = spc._create_model, verbose=0)

    # grid search hypers
    param_grid = {
        'epochs': [50],
        'batch_size': [64],
        'filters': [320],
        'kernel_size': [5],
        'conv_activation': ['tanh'],
        'conv_l2_regularizer': [0.01],
        'dropout_rate': [0.6],
        'dense_activation': ['relu', 'tanh'],
        'dense_l2_regularizer': [0.01],
        'activation': ['softmax'],
        'optimizer': ['nadam'],
        'loss_function': ['categorical_crossentropy']
    }

    # train
    if IS_FIT:
        IS_REFIT = False
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, refit=IS_REFIT, scoring=['f1_macro', 'precision_macro', 'recall_macro'], verbose=2)
        grid_result = grid.fit(X, y)
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        print(grid_result.cv_results_.keys())
        means = [grid_result.cv_results_['mean_test_f1_macro'], grid_result.cv_results_['mean_test_precision_macro'], grid_result.cv_results_['mean_test_recall_macro']]
        stds = [grid_result.cv_results_['std_test_f1_macro'], grid_result.cv_results_['std_test_precision_macro'], grid_result.cv_results_['std_test_recall_macro']]
        params = grid_result.best_params_
        for mean, stdev in zip(means, stds):
            print("\n{} ({})".format(mean, stdev))
        print("with:", params)
        if IS_REFIT:
            grid.best_estimator_.model.save('best')

if __name__ == "__main__":
    main()