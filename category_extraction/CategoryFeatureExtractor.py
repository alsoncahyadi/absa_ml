import sys

try:
    sys.path.insert(0, '.')
    from constants import Const
    sys.path.insert(0, Const.ROOT)
except:
    sys.path.insert(0, '..')
    from constants import Const

import dill
from we.cluster.KMeans import transform
from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import load_model

class CategoryFeatureExtractor (BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        cnn_model = load_model('model/cnn/CNN.model')
        features = {}
        y_pred = cnn_model.predict(X)
        
        #
        features['cnn_probability'] = y_pred
        
        #
        features['review'] = []
        for datum in X:
            features['review'].append(" ".join([str(token) for token in datum if token != 0]))
        
        #
        cluster_list = None
        with open('../we/cluster/cluster_list_1000.pkl', 'rb') as fi:
            cluster_list = dill.load(fi)
        X_cluster = transform(X, cluster_list)
        features['review_cluster'] = []
        for datum in X_cluster:
            features['review_cluster'].append(" ".join([str(token) for token in datum if token != 0]))
        
        return features