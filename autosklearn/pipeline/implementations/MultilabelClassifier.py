import warnings
import numpy as np

from sklearn.base import clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier, _ConstantPredictor


def _fit_binary(estimator, X, y, classes=None, sample_weight=None):
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
        estimator.fit(X, y, sample_weight=None)
    return estimator


class MultilabelClassifier(OneVsRestClassifier):
    """Subclasses sklearn.multiclass.OneVsRestClassifier in order to add
    sample weights. Works as original code, but forwards sample_weihts to
    base estimator

    Taken from:
    https://github.com/scikit-learn/scikit-learn/blob/a95203b/sklearn/multiclass.py#L203
    """

    def fit(self, X, y, sample_weight=None):
        """Fit underlying estimators.
        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.
        y : (sparse) array-like, shape = [n_samples] or [n_samples, n_classes]
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
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        columns = (col.toarray().ravel() for col in Y.T)
        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.
        self.estimators_ = [_fit_binary(estimator=self.estimator,
                                        X=X, y=column,
                                        classes=["not %s" % self.label_binarizer_.classes_[i], self.label_binarizer_.classes_[i]],
                                        sample_weight=sample_weight)
                            for i, column in enumerate(columns)]

        return self

