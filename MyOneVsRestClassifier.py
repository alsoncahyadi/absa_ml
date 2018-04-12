import array
import numpy as np
import warnings
import scipy.sparse as sp
import itertools

from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.base import MetaEstimatorMixin, is_regressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import (_check_partial_fit_first_call,
                               check_classification_targets,
                               _ovr_decision_function)
from sklearn.utils.metaestimators import _safe_split, if_delegate_has_method

from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from sklearn.externals.six.moves import zip as izip

def _fit_binary(estimator, X, y, classes=None):
    """Fit a single binary estimator."""
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not None:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn("Label %s is present in all training examples." %
                          str(classes[c]))
        estimator = _ConstantPredictor().fit(X, unique_y)
    else:
        estimator = clone(estimator)
        estimator.fit(X, y)
    return estimator


def _partial_fit_binary(estimator, X, y):
    """Partially fit a single binary estimator."""
    estimator.partial_fit(X, y, np.array((0, 1)))
    return estimator


def _predict_binary(estimator, X):
    """Make predictions using a single binary estimator."""
    if is_regressor(estimator):
        return estimator.predict(X)
        # probabilities of the positive class
    score = estimator.predict_proba(X)[:, 1]
    return score


def _check_estimator(estimator):
    """Make sure that an estimator implements the necessary methods."""
    if (not hasattr(estimator, "decision_function") and
            not hasattr(estimator, "predict_proba")):
        raise ValueError("The base estimator should implement "
                         "decision_function or predict_proba!")


class _ConstantPredictor(BaseEstimator):

    def fit(self, X, y):
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self, 'y_')

        return np.repeat(self.y_, X.shape[0])

    def decision_function(self, X):
        check_is_fitted(self, 'y_')

        return np.repeat(self.y_, X.shape[0])

    def predict_proba(self, X):
        check_is_fitted(self, 'y_')

        return np.repeat([np.hstack([1 - self.y_, self.y_])],
                         X.shape[0], axis=0)

class MyOneVsRestClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """One-vs-the-rest (OvR) multiclass/multilabel strategy

    Also known as one-vs-all, this strategy consists in fitting one classifier
    per class. For each classifier, the class is fitted against all the other
    classes. In addition to its computational efficiency (only `n_classes`
    classifiers are needed), one advantage of this approach is its
    interpretability. Since each class is represented by one and one classifier
    only, it is possible to gain knowledge about the class by inspecting its
    corresponding classifier. This is the most commonly used strategy for
    multiclass classification and is a fair default choice.

    This strategy can also be used for multilabel learning, where a classifier
    is used to predict multiple labels for instance, by fitting on a 2-d matrix
    in which cell [i, j] is 1 if sample i has label j and 0 otherwise.

    In the multilabel learning literature, OvR is also known as the binary
    relevance method.

    Read more in the :ref:`User Guide <ovr_classification>`.

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and one of `decision_function`
        or `predict_proba`.

    n_jobs : int, optional, default: 1
        The number of jobs to use for the computation. If -1 all CPUs are used.
        If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.

    Attributes
    ----------
    estimators_ : list of `n_classes` estimators
        Estimators used for predictions.

    classes_ : array, shape = [`n_classes`]
        Class labels.
    label_binarizer_ : LabelBinarizer object
        Object used to transform multiclass labels to binary labels and
        vice-versa.
    multilabel_ : boolean
        Whether a OneVsRestClassifier is a multilabel classifier.
    """
    def __init__(self, estimator, n_jobs=1):
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        Returns
        -------
        self
        """
        # A sparse LabelBinarizer, with sparse_output=True, has been shown to
        # outpreform or match a dense label binarizer in all cases and has also
        # resulted in less or equal memory consumption in the fit_ovr function
        # overall.
        from keras import backend as K
        K.clear_session()
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(delayed(_fit_binary)(
            self.estimator, X, column, classes=[
                "not %s" % self.label_binarizer_.classes_[i],
                self.label_binarizer_.classes_[i]])
            for i, column in enumerate(columns))

        return self

    @if_delegate_has_method('estimator')
    def partial_fit(self, X, y, classes=None):
        """Partially fit underlying estimators

        Should be used when memory is inefficient to train all data.
        Chunks of data can be passed in several iteration.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        classes : array, shape (n_classes, )
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is only required in the first call of partial_fit
            and can be omitted in the subsequent calls.

        Returns
        -------
        self
        """
        if _check_partial_fit_first_call(self, classes):
            if not hasattr(self.estimator, "partial_fit"):
                raise ValueError(("Base estimator {0}, doesn't have "
                                 "partial_fit method").format(self.estimator))
            self.estimators_ = [clone(self.estimator) for _ in range
                                (self.n_classes_)]

            # A sparse LabelBinarizer, with sparse_output=True, has been
            # shown to outperform or match a dense label binarizer in all
            # cases and has also resulted in less or equal memory consumption
            # in the fit_ovr function overall.
            self.label_binarizer_ = LabelBinarizer(sparse_output=True)
            self.label_binarizer_.fit(self.classes_)

        if len(np.setdiff1d(y, self.classes_)):
            raise ValueError(("Mini-batch contains {0} while classes " +
                             "must be subset of {1}").format(np.unique(y),
                                                             self.classes_))

        Y = self.label_binarizer_.transform(y)
        Y = Y.tocsc()
        columns = (col.toarray().ravel() for col in Y.T)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit_binary)(estimator, X, column)
            for estimator, column in izip(self.estimators_, columns))

        return self

    def predict(self, X):
        """Predict multi-class targets using underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        Returns
        -------
        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes].
            Predicted multi-class targets.
        """
        check_is_fitted(self, 'estimators_')
        if (hasattr(self.estimators_[0], "decision_function") and
                is_classifier(self.estimators_[0])):
            thresh = 0
        else:
            thresh = .2

        n_samples = _num_samples(X)
        if self.label_binarizer_.y_type_ == "multiclass":
            maxima = np.empty(n_samples, dtype=float)
            maxima.fill(-np.inf)
            argmaxima = np.zeros(n_samples, dtype=int)
            for i, e in enumerate(self.estimators_):
                pred = _predict_binary(e, X)
                np.maximum(maxima, pred, out=maxima)
                argmaxima[maxima == pred] = i
            return self.classes_[np.array(argmaxima.T)]
        else:
            indices = array.array('i')
            indptr = array.array('i', [0])
            for e in self.estimators_:
                indices.extend(np.where(_predict_binary(e, X) > thresh)[0])
                indptr.append(len(indices))
            data = np.ones(len(indices), dtype=int)
            indicator = sp.csc_matrix((data, indices, indptr),
                                      shape=(n_samples, len(self.estimators_)))
            return self.label_binarizer_.inverse_transform(indicator)

    @if_delegate_has_method(['_first_estimator', 'estimator'])
    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by label of classes.

        Note that in the multilabel case, each sample can have any number of
        labels. This returns the marginal probability that the given sample has
        the label in question. For example, it is entirely consistent that two
        labels both have a 90% probability of applying to a given sample.

        In the single label multiclass case, the rows of the returned matrix
        sum to 1.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : (sparse) array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.
        """
        check_is_fitted(self, 'estimators_')
        # Y[i, j] gives the probability that sample i has the label j.
        # In the multi-label case, these are not disjoint.
        Y = np.array([e.predict_proba(X)[:, 1] for e in self.estimators_]).T

        if len(self.estimators_) == 1:
            # Only one estimator, but we still want to return probabilities
            # for two classes.
            Y = np.concatenate(((1 - Y), Y), axis=1)

        if not self.multilabel_:
            # Then, probabilities should be normalized to 1.
            Y /= np.sum(Y, axis=1)[:, np.newaxis]
        return Y

    @if_delegate_has_method(['_first_estimator', 'estimator'])
    def decision_function(self, X):
        """Returns the distance of each sample from the decision boundary for
        each class. This can only be used with estimators which implement the
        decision_function method.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
        """
        check_is_fitted(self, 'estimators_')
        if len(self.estimators_) == 1:
            return self.estimators_[0].decision_function(X)
        return np.array([est.decision_function(X).ravel()
                         for est in self.estimators_]).T

    @property
    def multilabel_(self):
        """Whether this is a multilabel classifier"""
        return self.label_binarizer_.y_type_.startswith('multilabel')

    @property
    def n_classes_(self):
        return len(self.classes_)

    @property
    def coef_(self):
        check_is_fitted(self, 'estimators_')
        if not hasattr(self.estimators_[0], "coef_"):
            raise AttributeError(
                "Base estimator doesn't have a coef_ attribute.")
        coefs = [e.coef_ for e in self.estimators_]
        if sp.issparse(coefs[0]):
            return sp.vstack(coefs)
        return np.vstack(coefs)

    @property
    def intercept_(self):
        check_is_fitted(self, 'estimators_')
        if not hasattr(self.estimators_[0], "intercept_"):
            raise AttributeError(
                "Base estimator doesn't have an intercept_ attribute.")
        return np.array([e.intercept_.ravel() for e in self.estimators_])

    @property
    def _pairwise(self):
        """Indicate if wrapped estimator is using a precomputed Gram matrix"""
        return getattr(self.estimator, "_pairwise", False)

    @property
    def _first_estimator(self):
        return self.estimators_[0]
