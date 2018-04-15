param_grid = {
        'epochs': [25, 50],
        'batch_size': [64],
        'recurrent_dropout': [0.6, 0.2],
        'dropout_rate': [0.6],
        'dense_activation': ['tanh', 'relu'],
        'dense_l2_regularizer': [0.01],
        'activation': ['sigmoid'],
        'optimizer': ["nadam"],
        'loss_function': ['binary_crossentropy'],
        'gru_units': [256, 16],
        'units': [256, 16]
    }

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
sys.path.insert(0, '..')

import utils

from MyClassifier import MyClassifier, MultilabelKerasClassifier, KerasClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from keras import backend as K
from keras.models import Sequential, Input, Model, load_model
from keras.layers.convolutional import Conv1D
from keras.layers import Dense, LSTM, Dropout, Lambda, Bidirectional, TimeDistributed, RepeatVector, RNN, GRU, CuDNNGRU, CuDNNLSTM
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

class RNNOpinionTargetExtractor (MyClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.WEIGHTS_PATH = 'model/brnn/weights/BRNN.hdf5'
        self.MODEL_PATH = 'model/brnn/BRNN.model'
        self.WE_PATH = '../we/embedding_matrix.pkl'
       
        self.layer_embedding = self._load_embedding(self.WE_PATH, trainable=True, vocabulary_size=15000, embedding_vector_length=500)
        self.rnn_model = None
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def fit(self, X, y, **kwargs):
        self.rnn_model = self._create_model()
        self.rnn_model.save(self.MODEL_PATH)
        self.rnn_model.summary()
        mode = kwargs.get('mode', 'train_validate_split')
        if mode == "train_validate_split":
            self.rnn_model.fit(
                X, y,
                **kwargs
            )
            self.rnn_model.load_weights(self.WEIGHTS_PATH)
    
    def predict(self, X, **kwargs):
        y_pred = self.rnn_model.predict(X)
        return y_pred
    
    def score(self, X, y, **kwargs):
        if self.rnn_model != None:
            rnn_model = self.rnn_model
        else:
            rnn_model = self._create_model()
        rnn_model.load_weights(self.WEIGHTS_PATH)
        scores = rnn_model.evaluate(X, y, verbose=0)
        print("Test Set Accuracy: %.2f%%" % (scores[1]*100))
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
            for i, y_sents in enumerate(y):
                for y_tokens in y_sents[:end[i]]:
                    tmp.append(y_tokens)
            tmp = np.array(tmp)
            return tmp

        y_pred_raw = rnn_model.predict(X)
        y_pred = []

        for y_pred_raw_sents in y_pred_raw:
            y_pred_sents = []
            for y_pred_raw_tokens in y_pred_raw_sents:
                max_i = max_index(y_pred_raw_tokens)
                y = [0.] * 3 #number of classes to be predicted
                y[max_i] = 1.
                y_pred_sents.append(y)
            y_pred.append(y_pred_sents)
        y_pred = np.array(y_pred)
        # y_pred = np.argmax(get_decreased_dimension(y_pred_raw), axis=-1)
        # y_test = np.argmax(get_decreased_dimension(y_test), axis=-1)
        
        end = get_sentence_end_index(X)
        y_pred = get_decreased_dimension(y_pred, end)
        y_test = get_decreased_dimension(y_test, end)
        print(y_test.shape, y_pred.shape)

        from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
        AVERAGE = None
        print("F1-Score  : {}".format(f1_score(y_test, y_pred, average=AVERAGE)))
        print("Precision : {}".format(precision_score(y_test, y_pred, average=AVERAGE)))
        print("Recall    : {}".format(recall_score(y_test, y_pred, average=AVERAGE)))

        f1_score_macro = f1_score(y_test, y_pred, average='macro')
        print("F1-Score-Macro:", f1_score_macro)

        y_test = np.array(y_test)
        y_pred = np.array(y_pred)

        print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))

        is_show_confusion_matrix = kwargs.get('show_confusion_matrix', False)
        if is_show_confusion_matrix:
            self.plot_all_confusion_matrix(y_test, y_pred)
        
        return f1_score_macro

    def _fit_train_validate_split(self, X, y):
        pass

    def _fit_gridsearch_cv(self, X, y, param_grid, **kwargs):
        from sklearn.model_selection import GridSearchCV
        # y = np.argmax(y, axis=2)
        # print(y)
        np.random.seed(7)
        # Wrap in sklearn wrapper
        model = KerasClassifier(build_fn = self._create_model, verbose=0)
        # train
        IS_REFIT = kwargs.get('is_refit', 'f1_macro')
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, refit=IS_REFIT, verbose=1, scoring=['f1_macro', 'precision_macro', 'recall_macro'])
        grid_result = grid.fit(X, y)
        print(grid_result.cv_results_.keys())
        means = [grid_result.cv_results_['mean_test_f1_macro'], grid_result.cv_results_['mean_test_precision_macro'], grid_result.cv_results_['mean_test_recall_macro']]
        stds = [grid_result.cv_results_['std_test_f1_macro'], grid_result.cv_results_['std_test_precision_macro'], grid_result.cv_results_['std_test_recall_macro']]
        for mean, stdev in zip(means, stds):
            print("\n{} ({})".format(mean, stdev))
        params = grid_result.best_params_
        print("with:", params)
        with open('output/gridsearch_lstm.pkl', 'wb') as fo:
            dill.dump(grid_result.cv_results_, fo)
        if IS_REFIT:
            grid.best_estimator_.model.save('model/cnn/best.model')

    def _create_model(
        self,
        recurrent_dropout = 0.5,
        dropout_rate = 0.6,
        dense_activation = 'tanh',
        dense_l2_regularizer = 0.01,
        activation = 'sigmoid',
        optimizer = "nadam",
        loss_function = 'binary_crossentropy',
        threshold = 0.7,
        gru_units = 256,
        units = 256,
        

        **kwargs
    ):
        K.clear_session()
        MAX_SEQUENCE_LENGTH = kwargs.get("max_sequence_length")
        # Define Architecture
        layer_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        layer_embedding = self.layer_embedding(layer_input)
        layer_blstm = Bidirectional(LSTM(gru_units, return_sequences=True, recurrent_dropout=recurrent_dropout, stateful=False))(layer_embedding)
        # layer_blstm = Bidirectional(GRU(gru_units, return_sequences=True, recurrent_dropout=recurrent_dropout, stateful=False))(layer_embedding)
        layer_dropout_1 = TimeDistributed(Dropout(0.5, seed=7))(layer_blstm)
        layer_dense_1 = TimeDistributed(Dense(units, activation=dense_activation, kernel_regularizer=regularizers.l2(dense_l2_regularizer)))(layer_dropout_1)
        layer_softmax = TimeDistributed(Dense(3, activation=activation))(layer_dense_1)
        rnn_model = Model(inputs=layer_input, outputs=layer_softmax)

        # Create Optimizer
        optimizer = optimizers.SGD(lr=0.05, momentum=0.9, decay=0.0, nesterov=True)
        rnn_model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'],
                        sample_weight_mode="temporal")
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

def get_sentence_end_index(X):
    end = []
    for datum in X:
        for i, token in enumerate(datum):
            if token == -1:
                end.append(i)
                break
    return np.array(end)

def main():
    import numpy as np

    """
        Initialize data
    """
    X, y, X_test, y_test = utils.get_ote_dataset()
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.15, random_state=7)
    
    """
        Calculate Sample Weight
    """
    sample_weight = utils.get_sample_weight(X_train, y_train)
    
    """
        Make and fit the model
    """
    np.random.seed(7)

    ote = RNNOpinionTargetExtractor()

    n_epoch = 25
    # for i in range(n_epoch):
    #     print('Epoch #', i)
    #     model.fit(x=X_train, y=y_train, epochs=1, batch_size=32,
    #           validation_data=(X_validate, y_validate), callbacks=[checkpointer]
    #           ,sample_weight=sample_weight)
    #     model.reset_states()

    # checkpointer = ModelCheckpoint(filepath='model/brnn/weights/BRNN.hdf5', verbose=1, save_best_only=True)
    # ote.fit(X_train, y_train, epochs=n_epoch, batch_size=32,
    #     validation_data=(X_validate, y_validate), callbacks=[checkpointer]
    #     ,sample_weight=sample_weight)
    
    ote._fit_gridsearch_cv(X, y, param_grid)

    """
        Load best estimator and score it
    """
    best_model = load_model('model/cnn/best.model')
    del ote.rnn_model
    ote.rnn_model = best_model
    ote.score(X_test, y_test)
    
if __name__ == "__main__":
    main()