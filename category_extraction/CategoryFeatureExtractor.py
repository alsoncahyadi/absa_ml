from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import load_model

class CategoryFeatureExtractor (BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        cnn_model = load_model('model/cnn/best.model')
        features = {}
        y_pred = cnn_model.predict(X)
        features['cnn_probability'] = y_pred
        features['review'] = []
        for datum in X:
            features['review'].append(" ".join([str(token) for token in datum if token != 0]))
            
        return features