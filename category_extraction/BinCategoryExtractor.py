import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
sys.path.insert(0, '..')
sys.setrecursionlimit(999999999)

import utils
from ItemSelector import ItemSelector

from MyClassifier import MyClassifier, MultilabelKerasClassifier, KerasClassifier
from MyOneVsRestClassifier import MyOneVsRestClassifier
from CategoryFeatureExtractor import CategoryFeatureExtractor

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
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.neural_network import MLPClassifier

N_EPOCHS = 35
N_CV = 5


class BinCategoryExtractor (MyClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.WEIGHTS_PATH = 'model/ann/weights/ANN.hdf5'
        self.MODEL_PATH = 'model/ann/ANN.model'
        self.WE_PATH = '../we/embedding_matrix.pkl'
        self.COUNT_VECTORIZER_VOCAB_PATH = 'data/count_vectorizer_vocabulary.pkl'
       
        self.layer_embedding = self._load_embedding(self.WE_PATH, trainable=True, vocabulary_size=15000, embedding_vector_length=500)
        # for key, value in kwargs.items():
        #     setattr(self, key, value)
        count_vectorizer_vocab = None
        with open(self.COUNT_VECTORIZER_VOCAB_PATH, 'rb') as fi:
            count_vectorizer_vocab = dill.load(fi)

        self.pipeline = Pipeline([
            ('data', CategoryFeatureExtractor()),
            (
                'features', FeatureUnion(
                    transformer_list= [
                        ('cnn_probability', ItemSelector(key='cnn_probability')),
                        ('bag_of_ngram', Pipeline([
                            ('selector', ItemSelector(key='review')),
                            ('ngram', CountVectorizer(ngram_range=(1, 2), vocabulary=count_vectorizer_vocab)),
                        ]))
                    ]
                )
            ),
            # ('clf', MLPClassifier(hidden_layer_sizes=(128,), activation='tanh', solver='adam', batch_size=32, max_iter=25, verbose=1))
            ('clf', MyOneVsRestClassifier(KerasClassifier(build_fn = self._create_ann_model, verbose=0, epochs=N_EPOCHS), thresh=0.8))
        ])

    def _create_ann_model(
        self,
        dropout_rate = 0.6,
        dense_activation = 'tanh',
        dense_l2_regularizer = 0.01,
        activation = 'sigmoid',
        optimizer = "nadam",
        loss_function = 'binary_crossentropy',
        threshold = 0.75, #not used

        **kwargs
    ):
        n_cnn_proba = 4
        n_bag_of_bigrams = 8016

        total_inputs = n_bag_of_bigrams + n_cnn_proba

        INPUT_DIM = kwargs.get('input_dim', total_inputs)

        # Define Architecture
        layer_input = Input(shape=(INPUT_DIM,))
        layer_dense_1 = Dense(128, activation=dense_activation, kernel_regularizer=regularizers.l2(dense_l2_regularizer))(layer_input)
        layer_dropout_1 = Dropout(dropout_rate, seed=7)(layer_dense_1)
        layer_softmax = Dense(1, activation=activation)(layer_dropout_1)
        
        # Create Model
        ann_model = Model(inputs=layer_input, outputs=layer_softmax)
        
        # Compile
        ann_model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
        return ann_model

    
    def fit(self, X, y):
        self.pipeline.fit(X, y)
    
    def predict(self, X):
        return self.pipeline.predict(X)

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

    def _fit_gridsearch_cv(self, X, y, param_grid, **kwargs):
        from sklearn.model_selection import GridSearchCV
        np.random.seed(7)

        # train
        IS_REFIT = kwargs.get('is_refit', 'f1_macro')
        grid = GridSearchCV(estimator=self.pipeline, param_grid=param_grid, cv=N_CV, refit=IS_REFIT, verbose=1, scoring=['f1_macro', 'precision_macro', 'recall_macro'])
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
            del self.pipeline
            self.pipeline = grid.best_estimator_

    def load_estimators(self, n_estimators = 4, load_path='model/ann/best_{}.model'):
        estimators = []
        for i in range(n_estimators):
            ann_model = load_model(load_path.format(i))
            new_estimator = KerasClassifier(build_fn=self._create_ann_model, verbose=0, epochs=N_EPOCHS)
            new_estimator.model = ann_model
            estimators.append(new_estimator)
        ann_sklearn_model_index = len(self.pipeline.steps) - 1
        self.pipeline.steps[ann_sklearn_model_index][1].estimators_ = estimators
        return estimators

    def save_estimators(self, save_path='model/ann/best_{}.model'):
        ann_sklearn_model_index = len(self.pipeline.steps) - 1
        estimators = self.pipeline.steps[ann_sklearn_model_index][1].estimators_
        for i, estimator in enumerate(estimators):
            estimator.model.save(save_path.format(i))


def make_new_count_vectorizer_vocab():
    X, y, X_test, y_test = utils.get_ce_dataset()
    review = []
    for datum in X:
        review.append(" ".join([str(token) for token in datum if token != 0]))

    cv = CountVectorizer(ngram_range=(1, 2))
    cv.fit(review)
    vocab = cv.vocabulary_
    with open('data/count_vectorizer_vocabulary.pkl', 'wb') as fo:
        dill.dump(vocab, fo)

def binary():
    """
        Initialize data
    """
    X, y, X_test, y_test = utils.get_ce_dataset()
    
    """
        Make the model
    """
    np.random.seed(7)
    bi = BinCategoryExtractor()
    print(X.shape, y.shape)
    param_grid = {
        
    }
    bi._fit_gridsearch_cv(X, y, param_grid)
    # bi._fit_gridsearch_cv(X, y, param_grid)
    bi.score(X_test, y_test)
    #save the best OneVsRest model (ovr = OneVsRest)
    bi.save_estimators()
    print("DONE SAVING")
    new_estimators = bi.load_estimators()
    print("DONE LOADING")
    bi.score(X_test, y_test)

if __name__ == "__main__":
    binary()