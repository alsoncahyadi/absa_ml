from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import dill
from keras import Sequential
from keras.layers.embeddings import Embedding
from keras.wrappers.scikit_learn import BaseWrapper
from keras.utils import to_categorical
from keras.models import Model
from keras.callbacks import ModelCheckpoint

class MyModel (Model):
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
        validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
        sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, **kwargs):

        best_weights_path = kwargs.get('best_weights_path', '/tmp/weights.hdf5')
        
        checkpointer = ModelCheckpoint(filepath=best_weights_path, verbose=1, save_best_only=True)

        super(MyModel, self).fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[checkpointer],
            validation_split=validation_split, validation_data=validation_data, shuffle=shuffle, class_weight=class_weight,
            sample_weight=sample_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

        if (validation_split != 0.0) or (validation_data != None):
            self.load_weights(best_weights_path)
            if verbose == 1:
                print("Loaded best weight from", best_weights_path, "\n")

class MyClassifier (BaseEstimator, ClassifierMixin, object):
    def __init__ (self, **kwargs):
        # Make Tokenizer (load or from dataset)
        with open('../we/tokenizer.pkl', 'rb') as fi:
            self.tokenizer = dill.load(fi)
        self.kwargs = kwargs
        self.target_names = None
        self.VOCABULARY_SIZE = min(98806, kwargs.get('vocabulary_size', 15000))
        self.EMBEDDING_VECTOR_LENGTH = kwargs.get('embedding_vector_length', 500)

    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pass
    
    def score(self, X, y, verbose=1, **kwargs):
        y_pred = self.predict(X)
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
        return scores

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

class MultilabelKerasClassifier(BaseWrapper):
    """Implementation of the scikit-learn classifier API for Keras.
    """
    def __init__(self, build_fn=None, **sk_params):
        self.THRESHOLD = sk_params.get('threshold', 0.75)
        super(MultilabelKerasClassifier, self).__init__(build_fn, **sk_params)

    def fit(self, x, y, sample_weight=None, **kwargs):
        """Constructs a new model with `build_fn` & fit the model to `(x, y)`.

        # Arguments
            x : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit`

        # Returns
            history : object
                details about the training history at each epoch.

        # Raises
            ValueError: In case of invalid shape for `y` argument.
        """
        y = np.array(y)
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)
        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))
        self.n_classes_ = len(self.classes_)
        if sample_weight is not None:
            kwargs['sample_weight'] = sample_weight
        return super(MultilabelKerasClassifier, self).fit(x, y, **kwargs)

    def predict(self, x, **kwargs):
        """Returns the class predictions for the given test data.

        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments
                of `Sequential.predict_classes`.

        # Returns
            preds: array-like, shape `(n_samples,)`
                Class predictions.
        """
        kwargs = self.filter_sk_params(Sequential.predict_classes, kwargs)
        proba = self.model.predict(x, **kwargs)
        proba[proba >= self.THRESHOLD] = 1.
        proba[proba < self.THRESHOLD] = 0.
        return proba
        # if proba.shape[-1] > 1:
        #     classes = proba.argmax(axis=-1)
        # else:
        #     classes = (proba > 0.5).astype('int32')
        # return self.classes_[classes]

    def predict_proba(self, x, **kwargs):
        """Returns class probability estimates for the given test data.

        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments
                of `Sequential.predict_classes`.

        # Returns
            proba: array-like, shape `(n_samples, n_outputs)`
                Class probability estimates.
                In the case of binary classification,
                to match the scikit-learn API,
                will return an array of shape `(n_samples, 2)`
                (instead of `(n_sample, 1)` as in Keras).
        """
        kwargs = self.filter_sk_params(Sequential.predict_proba, kwargs)
        probs = self.model.predict(x, **kwargs)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs

    def score(self, x, y, **kwargs):
        """Returns the mean accuracy on the given test data and labels.

        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y: array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.evaluate`.

        # Returns
            score: float
                Mean accuracy of predictions on `x` wrt. `y`.

        # Raises
            ValueError: If the underlying model isn't configured to
                compute accuracy. You should pass `metrics=["accuracy"]` to
                the `.compile()` method of the model.
        """
        y = np.searchsorted(self.classes_, y)
        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        outputs = self.model.evaluate(x, y, **kwargs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        for name, output in zip(self.model.metrics_names, outputs):
            if name == 'acc':
                return output
        raise ValueError('The model is not configured to compute accuracy. '
                         'You should pass `metrics=["accuracy"]` to '
                         'the `model.compile()` method.')


class KerasClassifier(BaseWrapper):
    """Implementation of the scikit-learn classifier API for Keras.
    """

    def __init__(self, build_fn=None, **sk_params):
        self.model = None
        self.THRESHOLD = sk_params.get('threshold', 0.2)
        super(KerasClassifier, self).__init__(build_fn, **sk_params)

    def fit(self, x, y, sample_weight=None, **kwargs):
        """Constructs a new model with `build_fn` & fit the model to `(x, y)`.

        # Arguments
            x : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit`

        # Returns
            history : object
                details about the training history at each epoch.

        # Raises
            ValueError: In case of invalid shape for `y` argument.
        """
        y = np.array(y)
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)
        elif len(y.shape) == 3 and y.shape[2] > 1:
            self.classes_ = np.arange(y.shape[2])
        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))
        self.n_classes_ = len(self.classes_)
        if sample_weight is not None:
            kwargs['sample_weight'] = sample_weight
        return super(KerasClassifier, self).fit(x, y, **kwargs)

    def predict(self, x, **kwargs):
        """Returns the class predictions for the given test data.

        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments
                of `Sequential.predict_classes`.

        # Returns
            preds: array-like, shape `(n_samples,)`
                Class predictions.
        """
        kwargs = self.filter_sk_params(Sequential.predict_classes, kwargs)

        proba = self.model.predict(x, **kwargs)
        if proba.shape[-1] > 1:
            classes = proba.argmax(axis=-1)
        else:
            classes = (proba > self.THRESHOLD).astype('int32')
        return self.classes_[classes]

    def predict_proba(self, x, **kwargs):
        """Returns class probability estimates for the given test data.

        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments
                of `Sequential.predict_classes`.

        # Returns
            proba: array-like, shape `(n_samples, n_outputs)`
                Class probability estimates.
                In the case of binary classification,
                to match the scikit-learn API,
                will return an array of shape `(n_samples, 2)`
                (instead of `(n_sample, 1)` as in Keras).
        """
        kwargs = self.filter_sk_params(Sequential.predict_proba, kwargs)
        probs = self.model.predict(x, **kwargs)

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs

    def score(self, x, y, **kwargs):
        """Returns the mean accuracy on the given test data and labels.

        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y: array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.evaluate`.

        # Returns
            score: float
                Mean accuracy of predictions on `x` wrt. `y`.

        # Raises
            ValueError: If the underlying model isn't configured to
                compute accuracy. You should pass `metrics=["accuracy"]` to
                the `.compile()` method of the model.
        """
        y = np.searchsorted(self.classes_, y)
        kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__
        if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
            y = to_categorical(y)

        outputs = self.model.evaluate(x, y, **kwargs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        for name, output in zip(self.model.metrics_names, outputs):
            if name == 'acc':
                return output
        raise ValueError('The model is not configured to compute accuracy. '
                         'You should pass `metrics=["accuracy"]` to '
                         'the `model.compile()` method.')
