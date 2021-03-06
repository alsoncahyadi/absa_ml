from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import numpy as np
import dill
import itertools
import abc, time
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.python.keras.wrappers.scikit_learn import BaseWrapper
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.callbacks import ModelCheckpoint
from constants import Const

class Model (KerasModel):
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=[],
        validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
        sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, **kwargs):

        best_weights_path = kwargs.get('best_weights_path', '/tmp/weights.hdf5')
        is_no_validation_data = (validation_split == 0.0) and (validation_data == None)
        if is_no_validation_data:
            monitor = 'loss'
        else:
            monitor = 'val_loss'

        checkpointer = ModelCheckpoint(filepath=best_weights_path, save_best_only=True, verbose=1, monitor=monitor)

        super(Model, self).fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[checkpointer] + callbacks,
            validation_split=validation_split, validation_data=validation_data, shuffle=shuffle, class_weight=class_weight,
            sample_weight=sample_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

        self.load_weights(best_weights_path)
        if verbose == 1:
            print("Loaded best weight from", best_weights_path, "\n")

class MyClassifier (BaseEstimator, ClassifierMixin, object):
    def __init__ (self, **kwargs):
        # Make Tokenizer (load or from dataset)
        with open(Const.TOKENIZER_PATH, 'rb') as fi:
            self.tokenizer = dill.load(fi)
        self.kwargs = kwargs
        self.target_names = []
        self.VOCABULARY_SIZE = min(98806, kwargs.get('vocabulary_size', Const.VOCABULARY_SIZE))
        self.EMBEDDING_VECTOR_LENGTH = kwargs.get('embedding_vector_length', Const.EMBEDDING_VECTOR_LENGTH)

    def fit(self, X, y, **kwargs):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y, verbose=1, **kwargs):
        y_pred = self.predict(X, **kwargs)
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

    def _load_embedding(self, path_to_embedding_matrix, **kwargs):

        #load the embedding matrix
        embedding_matrix = []
        with open(path_to_embedding_matrix, 'rb') as fi:
            embedding_matrix = dill.load(fi)

        input_dim = self.VOCABULARY_SIZE
        layer_embedding = Embedding(input_dim,
                                    self.EMBEDDING_VECTOR_LENGTH,
                                    weights=[embedding_matrix[:input_dim]],
                                    trainable=kwargs.get('trainable', False),
                                    mask_zero=True)
        return layer_embedding

    def _fit_new_gridsearch_cv(self, X, y, params, k=5, verbose=0, fit_verbose=0, score_verbose=0, thresholds=None,
        result_path="output/gridsearch_cv_result.csv", **kwargs):
        score_metrics = ['f1_macro', 'precision_macro', 'recall_macro']
        param_names, param_values = zip(*params)
        DELIM = ';'
        with open(result_path, 'w') as fo:
            fo.write(DELIM.join(score_metrics + list(param_names)) + '\n')

        param_value_combinations = list(itertools.product(*param_values))
        print("FITTING {}x FOR {} combinations, {} folds".format(len(param_value_combinations)*k, len(param_value_combinations), k))
        for i, current_param_value in enumerate(param_value_combinations):
            # Print log
            print('\n===========', i+1, '/', len(param_value_combinations), '===========')
            for key, value in zip(param_names, current_param_value):
                print(" ", key, ":", value)

            # Fit
            start_time = time.time()
            current_param = dict(zip(param_names, current_param_value))
            current_cv_score= self._fit_cv(X, y, k=k, verbose=fit_verbose, score_verbose=score_verbose, thresholds=thresholds, **current_param, **kwargs)

            # Write result
            with open(result_path, 'a') as fo:
                current_param_value_str = DELIM.join([str(v) for v in current_param_value])
                score_str = DELIM.join([str(current_cv_score[score_metric]) for score_metric in score_metrics])
                line = score_str + DELIM + current_param_value_str + "\n"
                fo.write(line)
            print("Done in: {0:.2f} s".format(time.time() - start_time))

    @abc.abstractmethod
    def set_threshold(self, thresh):
        raise NotImplementedError

    @abc.abstractmethod
    def get_threshold(self):
        raise NotImplementedError

    def _fit_cv(self, X, y, k=5, verbose=0, score_verbose=0, sample_weight=None, thresholds=None, keras_multiple_output=False, **kwargs):
        if type(sample_weight).__name__ != "NoneType":
            sample_weight_folds = np.array_split(sample_weight, k)

        if keras_multiple_output:
            X_folds = []
            for i in range(k):
                X_folds.append([])
            for X_single in X:
                X_single_folds = np.array_split(X_single, k)
                for i, X_single_fold in enumerate(X_single_folds):
                    X_folds[i].append(X_single_fold)
        else:
            X_folds = np.array_split(X, k)
        y_folds = np.array_split(y, k)

        num_classes = 2 if len(self.target_names) == 1 else len(self.target_names)

        precision_scores = [[] for _ in range(num_classes)]
        recall_scores = [[] for _ in range(num_classes)]
        f1_scores = [[] for _ in range(num_classes)]
        precision_means = []
        recall_means = []
        f1_means = []
        thresholds_used = []

        total_score_time = 0.

        for i in range(k):

            X_train = list(X_folds)
            X_test  = X_train.pop(i)
            if keras_multiple_output:
                X_train_new = []
                for j in range(len(X)):
                    X_train_new.append([])
                for X_train_fold in X_train:
                    for j, X_train_fold_single in enumerate(X_train_fold):
                        X_train_new[j] += list(X_train_fold_single)
                X_train = [np.array(X_train_new_single) for X_train_new_single in X_train_new]
            else:
                X_train = np.concatenate(X_train)

            y_train = list(y_folds)
            y_test  = y_train.pop(i)
            y_train = np.concatenate(y_train)

            sample_weight_train = None
            if type(sample_weight).__name__ != "NoneType":
                sample_weight_train = list(sample_weight_folds)
                sample_weight_train.pop(i)
                sample_weight_train = np.concatenate(sample_weight_train)

            if type(sample_weight_train).__name__ == "NoneType":
                self.fit(X_train, y_train
                    , verbose = verbose
                    , **kwargs
                )
            else:
                self.fit(X_train, y_train
                    , verbose = verbose
                    , **kwargs
                    , sample_weight = sample_weight_train
                )

            start_time = time.time()
            if thresholds != None:
                if len(thresholds) == 0:
                    raise ValueError('thresholds should not be empty')
                scoress = []
                f1_temp = []
                for j, threshold in enumerate(thresholds):
                    self.set_threshold(threshold)
                    scoress.append(self.score(X_test,y_test, verbose=0))
                    f1_temp.append(scoress[j]['f1_score_macro'])
                f1_temp = np.array(f1_temp)
                max_f1_idx = f1_temp.argmax()

                scores = scoress[max_f1_idx]
                thresholds_used.append(thresholds[max_f1_idx])
            else:
                included_features = kwargs.get('included_features', None)
                scores = self.score(X_test, y_test, verbose=0, included_features=included_features)
                try:
                    thresholds_used.append(self.get_threshold())
                except:
                    thresholds_used = None
            total_score_time += time.time() - start_time

            for j in range(num_classes):
                precision_scores[j].append(scores['precision_scores'][j])
                recall_scores[j].append(scores['recall_scores'][j])
                f1_scores[j].append(scores['f1_scores'][j])

        for i in range(num_classes):
            precision_mean = np.array(precision_scores[i]).mean()
            recall_mean = np.array(recall_scores[i]).mean()
            f1_mean = np.array(f1_scores[i]).mean()

            precision_means.append(precision_mean)
            recall_means.append(recall_mean)
            f1_means.append(f1_mean)

        precision_macro = np.array(precision_means).mean()
        recall_macro = np.array(recall_means).mean()
        f1_macro = np.array(f1_means).mean()

        if score_verbose > 0:
            print("\t>>> SCORE {0:.2f}s <<<".format(total_score_time))
            print("\tPrecision: {0:.5f}".format(precision_macro))
            print("\tRecall:    {0:.5f}".format(recall_macro))
            print("\tF1-score:  {0:.5f}".format(f1_macro))
            print("\tP -means:  {}".format(precision_means))
            print("\tR -means:  {}".format(recall_means))
            print("\tF1-means:  {}".format(f1_means))
            print("\tThresh  :  {}".format(thresholds_used))

        scores = {
            'precision_means': precision_means,
            'recall_means': recall_means,
            'f1_means': f1_means,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_scores': precision_scores,
            'recall_scores': recall_scores,
            'f1_scores': f1_scores,
            'thresholds_used': thresholds_used,
        }

        return scores

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
