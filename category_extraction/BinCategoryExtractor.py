params = [
    ("epochs", [50, 25]),
    ("dropout_rate", [0.5, 0., 0.25, 0.75]),
    ("dense_activation", ['tanh']),
    ("dense_l2_regularizer", [0.01, 0.]),
    ("activation", ['sigmoid', 'softmax']),
    ("optimizer", ["nadam"]),
    ("loss_function", ['binary_crossentropy', 'categorical_crossentropy']),
    ("threshold", [0.8]),
    ("included_features", [[0], [0,1], [0,2], [1,2], [1,2,3]])
]

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

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

N_CV = 5


class BinCategoryExtractor (MyClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.WEIGHTS_PATH = 'model/ann/weights/ANN.hdf5'
        self.MODEL_PATH = 'model/ann/ANN.model'
        self.WE_PATH = '../we/embedding_matrix.pkl'
        self.COUNT_VECTORIZER_VOCAB_PATH = 'data/count_vectorizer_vocabulary.pkl'
        self.COUNT_VECTORIZER_VOCAB_CLUSTER_PATH = 'data/count_vectorizer_vocabulary_cluster.pkl'
        self.target_names = ['food', 'service', 'price', 'place']
       
        self.layer_embedding = self._load_embedding(self.WE_PATH, trainable=True, vocabulary_size=15000, embedding_vector_length=500)
        # for key, value in kwargs.items():
        #     setattr(self, key, value)
        self.count_vectorizer_vocab = None
        with open(self.COUNT_VECTORIZER_VOCAB_PATH, 'rb') as fi:
            self.count_vectorizer_vocab = dill.load(fi)
        
        self.count_vectorizer_vocab_cluster = None
        with open(self.COUNT_VECTORIZER_VOCAB_CLUSTER_PATH, 'rb') as fi:
            self.count_vectorizer_vocab_cluster = dill.load(fi)

        print(len(self.count_vectorizer_vocab_cluster.items()))

        self.pipeline = Pipeline([
            ('data', CategoryFeatureExtractor()),
            (
                'features', FeatureUnion(
                    transformer_list= [
                        ('cnn_probability', ItemSelector(key='cnn_probability')),
                        ('bag_of_bigram', Pipeline([
                            ('selector', ItemSelector(key='review')),
                            ('ngram', CountVectorizer(ngram_range=(1, 2), vocabulary=self.count_vectorizer_vocab)),
                        ])),
                        ('bag_of_bigram_word_cluster', Pipeline([
                            ('selector', ItemSelector(key='review_cluster')),
                            ('ngram', CountVectorizer(ngram_range=(1, 2), vocabulary=self.count_vectorizer_vocab_cluster)),
                        ]))
                    ]
                )
            ),
            # ('clf', MLPClassifier(hidden_layer_sizes=(128,), activation='tanh', solver='adam', batch_size=32, max_iter=25, verbose=1))
            ('clf', MyOneVsRestClassifier(KerasClassifier(build_fn = self._create_ann_model, verbose=0, epochs=50)))
        ])

    def _create_ann_model(
        self,
        dropout_rate = 0.6,
        dense_activation = 'tanh',
        dense_l2_regularizer = 0.01,
        activation = 'sigmoid',
        optimizer = "nadam",
        loss_function = 'binary_crossentropy',
        included_features = [0,1,2],

        **kwargs
    ):
        n_cnn_proba = 4
        n_bag_of_bigrams = 8016
        n_bag_of_bigrams_cluster = 3755

        sums = [n_cnn_proba, n_bag_of_bigrams, n_bag_of_bigrams_cluster]
        included_features = included_features
        included_sums = []
        for included_feature in included_features:
            included_sums.append(sums[included_feature])
        sum_input = np.array(included_sums).sum()

        total_inputs = sum_input

        INPUT_DIM = kwargs.get('input_dim', total_inputs)

        # Define Architecture
        layer_input = Input(shape=(INPUT_DIM,))
        layer_dense_1 = Dense(64, activation=dense_activation, kernel_regularizer=regularizers.l2(dense_l2_regularizer))(layer_input)
        layer_dropout_1 = Dropout(dropout_rate, seed=7)(layer_dense_1)
        layer_softmax = Dense(1, activation=activation)(layer_dropout_1)
        
        # Create Model
        ann_model = Model(inputs=layer_input, outputs=layer_softmax)
        
        # Compile
        ann_model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
        return ann_model

    
    def fit(self, X, y,
            epochs = 50,
            batch_size = 64,
            dropout_rate = 0.6,
            dense_activation = 'tanh',
            dense_l2_regularizer = 0.01,
            activation = 'sigmoid',
            optimizer = "nadam",
            loss_function = 'binary_crossentropy',
            threshold = 0.5,
            included_features = [0, 1, 2],

            **kwargs
        ):
        transformer_list = [
            ('cnn_probability', ItemSelector(key='cnn_probability')),
            ('bag_of_bigram', Pipeline([
                ('selector', ItemSelector(key='review')),
                ('ngram', CountVectorizer(ngram_range=(1, 2), vocabulary=self.count_vectorizer_vocab)),
            ])),
            ('bag_of_bigram_word_cluster', Pipeline([
                ('selector', ItemSelector(key='review_cluster')),
                ('ngram', CountVectorizer(ngram_range=(1, 2), vocabulary=self.count_vectorizer_vocab_cluster)),
            ]))
        ]

        self.pipeline = Pipeline([
            ('data', CategoryFeatureExtractor()),
            (
                'features', FeatureUnion(
                    transformer_list= [transformer_list[included_feature] for included_feature in included_features]
                )
            ),
            ('clf', MyOneVsRestClassifier(KerasClassifier(
                build_fn = self._create_ann_model,
                verbose=0, epochs=epochs, batch_size=batch_size,
                dropout_rate = dropout_rate,
                dense_activation = dense_activation,
                dense_l2_regularizer = dense_l2_regularizer,
                activation = activation,
                optimizer = optimizer,
                loss_function = loss_function,
                included_features = included_features,
                ), thresh=threshold
            ))
        ])
        self.pipeline.fit(X, y)
    
    def predict(self, X):
        return self.pipeline.predict(X)

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
            self.save_estimators()

    def load_estimators(self, n_estimators = 4, load_path='model/ann/best_{}.model'):
        estimators = []
        for i in range(n_estimators):
            ann_model = load_model(load_path.format(i))
            new_estimator = KerasClassifier(build_fn=self._create_ann_model, verbose=0, epochs=50)
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

    def set_threshold(self, thresh):
        self.pipeline.steps[len(self.pipeline.steps)-1][1].thresh = thresh

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
    bi._fit_new_gridsearch_cv(X, y, params, verbose=1, fit_verbose=1, score_verbose=1, result_path='output/gridsearch_cv_result_bin.csv')
    # bi.fit(X, y, 
    #     epochs= 50,
    #     dropout_rate= 0.6,
    #     dense_activation= 'tanh',
    #     dense_l2_regularizer= 0.01,
    #     activation= 'sigmoid',
    #     optimizer= "nadam",
    #     loss_function= 'binary_crossentropy',
    #     threshold= 0.2,
    #     included_features= [0, 1],
    #     verbose = 1
    # )

    # bi.save_estimators()
    # bi.load_estimators()
    
    # thresh_to_try = [0.5, 0.55, 0.6, 0.65, 0.7, 0.725, 0.75, 0.775, 0.7875, 0.8, 0.825, 0.85, 0.875, 0.9]
    # for thresh in thresh_to_try:
    #     print("\nTHRESH: {}".format(thresh))
    #     bi.set_threshold(thresh); bi.score(X_test, y_test)

if __name__ == "__main__":
    import time
    start_time = time.time()
    binary()
    print("DONE IN {} seconds".format(time.time() - start_time))