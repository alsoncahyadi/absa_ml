import itertools
import os
import sys
import time

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

import sklearn_crfsuite
from collections import Counter
from threading import Thread
from sklearn_crfsuite import metrics
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.grid_search import RandomizedSearchCV

# from OpinionTargetFeatureExtractor import OpinionTargetFeatureExtractor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

sys.path.insert(0, '..')


NUM_THREAD = 4

class CRFOpinionTargetExtractor (MyClassifier):
    def __init__(self):
        self.crf_model = sklearn_crfsuite.CRF()
        self.target_names = ['O', 'ASPECT-B', 'ASPECT-I']
        self.MODEL_PATH = 'model/crf/CRF.model'
        super(CRFOpinionTargetExtractor).__init__()
    
    def fit(self, X, y,
        is_save=False,
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    ):
        self.crf_model = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=all_possible_transitions
        )
        self.crf_model.fit(X, y)

        if is_save:
            utils.save_object(crf, self.MODEL_PATH)

    def predict(self, X):
        y_pred = self.crf_model.predict(X)
        return y_pred
    
    def _print_transitions(self, trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

    def _print_top_transitions(self):
        print("\n\n >> Top likely transitions:")
        self._print_transitions(Counter(self.crf_model.transition_features_).most_common(5))
        print("\n >> Top unlikely transitions:")
        self._print_transitions(Counter(self.crf_model.transition_features_).most_common()[-5:])

    def _print_state_features(self, state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-8s %s" % (weight, label, attr))

    def _print_top_state_features(self):
        print("\n\n Top positive:")
        self._print_state_features(Counter(self.crf_model.state_features_).most_common(30))
        print("\n Top negative:")
        self._print_state_features(Counter(self.crf_model.state_features_).most_common()[-30:])

    def bio_classification_report(self, y_test, y_pred):
        labels = self.target_names
        # labels.remove('O')

        sorted_labels = sorted(
            labels,
            key=lambda name: (name[:1], name[len(name)-1])
        )
        print(sorted_labels)
        return metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, digits=4
        )

    def score(self, X, y, verbose=1, **kwargs):
        print("=========================================")        
        y_pred = self.predict(X, **kwargs)
        print(self.bio_classification_report(y, y_pred))
        self._print_top_state_features()
        self._print_top_transitions()
        print("=========================================")        
        from sklearn_crfsuite.utils import flatten
        for i in range(len(y)):
            if len(y[i]) != len(y_pred[i]):
                print(i, ')', len(y[i]), len(y_pred[i]))
                print(' '.join([word['0:word'] for word in X[i]]))

        y = flatten(y)
        y_pred = flatten(y_pred)

        f1_score_macro = f1_score(y, y_pred, average='macro')
        precision_score_macro = precision_score(y, y_pred, average='macro')
        recall_score_macro = recall_score(y, y_pred, average='macro')
        f1_scores = f1_score(y, y_pred, average=None)
        precision_scores = precision_score(y, y_pred, average=None)
        recall_scores = recall_score(y, y_pred, average=None)
        accuracy = accuracy_score(y, y_pred)

        scores = {
            'f1_score_macro': f1_score_macro,
            'precision_score_macro': precision_score_macro,
            'recall_score_macro': recall_score_macro,
            'f1_scores': f1_scores,
            'precision_scores': precision_scores,
            'recall_scores': recall_scores,
            'accuracy': accuracy
        }

        if verbose == 1:
            print("F1-Score  : {}".format(f1_scores))
            print("Precision : {}".format(precision_scores))
            print("Recall    : {}".format(recall_scores))
            print("Accuracy  : {}".format(accuracy))
            print("F1-Score-Macro:", f1_score_macro)
            print("P -Score-Macro:", precision_score_macro)
            print("R -Score-Macro:", recall_score_macro)
            print("Confusion Matrix:")
            try:
                print(confusion_matrix(y, y_pred))
            except:
                print("Can't be shown")
        return scores

    # def k_fold_validation(self, data_train, model_file_path, **kwargs):
    #     print("\n=== CROSS VALIDATING {} DATA WITH k: {} ===".format(len(data_train), kwargs.get('k', 4)))
    #     print(" > Extracting features")
    #     X_train, y_train = OpinionTargetFeatureExtractor.to_dataset(parsed_sentences=data_train, feature_detector=OpinionTargetFeatureExtractor.ner_features)
    #     print(" > Start training. . .")
    #     master_begin_time = time.time()
    #     crf = sklearn_crfsuite.CRF(
    #         algorithm='lbfgs',
    #         max_iterations=100,
    #         all_possible_transitions=True
    #     )
    #     params_space = {
    #         'c1': scipy.stats.expon(scale=0.5),
    #         'c2': scipy.stats.expon(scale=0.05),
    #     }

    #     labels = ['ASPECT-I', 'ASPECT-B']

    #     # use the same metric for evaluation
    #     f1_scorer = make_scorer(
    #         metrics.flat_f1_score,
    #         average='weighted', labels=labels
    #     )

    #     # search
    #     rs = RandomizedSearchCV(crf, params_space,
    #                             cv=kwargs.get('k', 4),
    #                             verbose=1,
    #                             n_jobs=-1,
    #                             n_iter=kwargs.get('n_iter', 10),
    #                             scoring=f1_scorer)
    #     rs.fit(X_train, y_train)
    #     master_end_time = time.time()
    #     print(" > Training done in {} seconds".format(master_end_time - master_begin_time))

    #     crf = rs.best_estimator_
    #     print(' >> best params:', rs.best_params_)
    #     print(' >> best CV score:', rs.best_score_)
    #     print(' >> model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
    #     utils.save_object(crf, model_file_path)
    #     print(" > Model saved in {}".format(model_file_path))

def crf():
    """
        Initialize data
    """
    X, y, X_test, y_test = utils.get_crf_ote_dataset()
    for idx, (i, j) in enumerate(zip(X, y)):
        if len(i) != len(j):
            print(idx, len(i), len(j))
            for k in range(min(len(i), len(j))):
                print('\t', i[k]['0:word'], '\t', j[k])
            print()
    crf_ote = CRFOpinionTargetExtractor()
    print("> fitting")
    crf_ote.fit(X, y)
    print("> scoring")
    crf_ote.score(X_test, y_test)

if __name__ == "__main__":
    utils.time_log(crf)