from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import dill
from keras.layers.embeddings import Embedding

class MyClassifier (BaseEstimator, ClassifierMixin, object):
    def __init__ (self, **kwargs):
        # Make Tokenizer (load or from dataset)
        with open('../we/tokenizer.pkl', 'rb') as fi:
            self.tokenizer = dill.load(fi)
        self.kwargs = kwargs
        self.VOCABULARY_SIZE = min(98806, kwargs.get('vocabulary_size', 15000))
        self.EMBEDDING_VECTOR_LENGTH = kwargs.get('embedding_vector_length', 500)

    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pass
    
    def score(self, X, y):
        pass

    def _load_embedding(self, path_to_embedding_matrix, **kwargs):

        #load the embedding matrix
        embedding_matrix = []
        with open(path_to_embedding_matrix, 'rb') as fi:
            embedding_matrix = dill.load(fi)

        layer_embedding = Embedding(self.VOCABULARY_SIZE,
                                    self.EMBEDDING_VECTOR_LENGTH,
                                    weights=[embedding_matrix[:self.VOCABULARY_SIZE]],
                                    trainable=kwargs.get('trainable', False))
        return layer_embedding