params = [
    ('algorithm', ['lbfgs']),
    ('max_iterations', [None]),
    ('c1', [1.0, 0.1, 0.01, 0.001,]),
    ('c2', [1.0, 0.1, 0.01, 0.001,]),
    ('included_features', [
        ['word', 'pos'],
        ['word', 'pos', 'cluster'],
        ['word', 'cluster'],
        ['cluster', 'pos'],
        ['rnn_proba', 'word', 'cluster'],
        ['rnn_proba', 'cluster', 'pos'],
        ['rnn_proba', 'word', 'pos'],
        ['rnn_proba', 'word', 'pos', 'cluster']
    ]),
    ('included_words', [
        [-2,-1,0,1,2],
        [-2,-1,0],
        [-1,0,1],
        [-1,0],
    ]),
]

import itertools
import os
import sys
import time

try:
    sys.path.insert(0, '.')
    from constants import Const
    sys.path.insert(0, Const.ROOT)
except:
    sys.path.insert(0, '..')
    from constants import Const

from OpinionTargetFeatureExtractor import extract_features
import dill
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (GRU, LSTM, RNN, Bidirectional,
                          Dense, Dropout, Lambda, RepeatVector,
                          TimeDistributed, Concatenate)
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import (AveragePooling1D, GlobalMaxPooling1D,
                                  MaxPooling1D)
from tensorflow.keras import Input, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence, text
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite.utils import flatten

# from OpinionTargetFeatureExtractor import OpinionTargetFeatureExtractor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, Const.ROOT)


NUM_THREAD = 4

class CRFOpinionTargetExtractor (MyClassifier):
    def __init__(self):
        self.crf_model = sklearn_crfsuite.CRF()
        self.target_names = ['O', 'ASPECT-B', 'ASPECT-I']
        self.MODEL_PATH = Const.OTE_ROOT + 'model/crf/CRF.model'
        super(CRFOpinionTargetExtractor).__init__()

    def fit(self, X, y,
        is_save=False,
        algorithm='lbfgs',
        max_iterations=None,
        all_possible_transitions=True,
        included_features = ['word'],
        included_words = [-2,-1,0,1,2],
        verbose=False,
        **kwargs
    ):
        X = extract_features(X, included_features = included_features, included_words=included_words)
        self.crf_model = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            max_iterations=max_iterations,
            all_possible_transitions=all_possible_transitions,
            verbose=verbose,
            **kwargs,
        )
        self.crf_model.fit(X, y)
        if is_save:
            utils.save_object(self.crf_model, self.MODEL_PATH)

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
        X = extract_features(X)
        y_pred = self.predict(X, **kwargs)
        if verbose == 2:
            print("=========================================")
            print(self.bio_classification_report(y, y_pred))
            self._print_top_state_features()
            self._print_top_transitions()
            print("=========================================")

        for i in range(len(y)):
            if len(y[i]) != len(y_pred[i]):
                print(i, ')', len(y[i]), len(y_pred[i]))
                print(' '.join([word['word'] for word in X[i]]))

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

        if verbose > 0:
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

    def load_best_model(self):
        best_model = None
        with open(Const.OTE_ROOT + "model/crf/best.model", 'rb') as fi:
            best_model = dill.load(fi)
        del self.crf_model
        self.crf_model = best_model

def crf():
    """
        Initialize data
    """

    included_features = [
        'rnn_proba',
        'word',
        'cluster',
        'pos',
    ]

    included_words = [-2,-1,0]

    print("> extracting features")
    X_pos, y, X_test_pos, y_test = utils.get_crf_ote_dataset()
    print(len(X_test_pos))

    # for idx, (i, j) in enumerate(zip(X, y)):
    #     if len(i) != len(j):
    #         print(idx, len(i), len(j))
    #         for k in range(min(len(i), len(j))):
    #             print('\t', i[k]['0:word'], '\t', j[k])
    #         print()
    crf_ote = CRFOpinionTargetExtractor()
    # """
    print("> fitting")

    """ GRIDSEARCH CV """
    # crf_ote._fit_new_gridsearch_cv(X_pos, y, params, score_verbose=True, result_path=Const.OTE_ROOT + 'output/gridsearch_cv_result_crf.csv')

    """ FIT """
    # crf_ote.fit(X_pos, y,
    #     algorithm='lbfgs',
    #     c1=0.01,
    #     c2=1.0,
    #     max_iterations=None,
    #     epsilon=1e-5,
    #     delta=1e-5,
    #     included_features=included_features,
    #     included_words=included_words,
    #     verbose=False,
    #     is_save=True,
    # )

    crf_ote.load_best_model()
    print("> scoring")
    crf_ote.score(X_test_pos, y_test, verbose=2)

def get_wrong_preds(data='train'):
    ote_crf = CRFOpinionTargetExtractor()
    ote_crf.load_best_model()

    X_pos, y, X_pos_test, y_test, df, df_test = utils.get_crf_ote_dataset(return_df = True)

    data = 'test'
    if data == 'test':
        df = df_test
        X_pos = X_pos_test
        y = y_test

    print(len(df))
    X = extract_features(X_pos)
    y_pred = ote_crf.predict(X)

    # ote_crf.score(X_pos, y)

    cnt = 0
    for i, (words, y_pred_single, y_single) in enumerate(list(zip(df['sentences'].tolist(), y_pred, y.tolist()))):
        is_wrong_word_present = False
        is_first_wrong_word = True
        for j, (word, y_pred_token, y_token) in enumerate(list(zip(words, y_pred_single, y_single))):
            if y_pred_token != y_token:
                cnt += 1
                is_wrong_word_present = True
                if is_first_wrong_word:
                    print("{})".format(i), " ".join(words))
                    is_first_wrong_word = False
                print(word, '\t | P:',y_pred_token, '\t| A:', y_token)
        if is_wrong_word_present:
            print()
    print(cnt, "words misclasified")

if __name__ == "__main__":
    # utils.time_log(get_wrong_preds)
    utils.time_log(crf)
