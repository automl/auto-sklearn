import resource

import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import *
from autosklearn.pipeline.implementations.util import softmax


# From the scikit-learn master branch. Will hopefully be there in sklearn 0.17
def _ovr_decision_function(predictions, confidences, n_classes):
    """Compute a continuous, tie-breaking ovr decision function.
    It is important to include a continuous value, not only votes,
    to make computing AUC or calibration meaningful.
    Parameters
    ----------
    predictions : array-like, shape (n_samples, n_classifiers)
        Predicted classes for each binary classifier.
    confidences : array-like, shape (n_samples, n_classifiers)
        Decision functions or predicted probabilities for positive class
        for each binary classifier.
    n_classes : int
        Number of classes. n_classifiers must be
        ``n_classes * (n_classes - 1 ) / 2``
    """
    n_samples = predictions.shape[0]
    votes = np.zeros((n_samples, n_classes))
    sum_of_confidences = np.zeros((n_samples, n_classes))

    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            sum_of_confidences[:, i] -= confidences[:, k]
            sum_of_confidences[:, j] += confidences[:, k]
            votes[predictions[:, k] == 0, i] += 1
            votes[predictions[:, k] == 1, j] += 1
            k += 1

    max_confidences = sum_of_confidences.max()
    min_confidences = sum_of_confidences.min()

    if max_confidences == min_confidences:
        return votes

    # Scale the sum_of_confidences to (-0.5, 0.5) and add it with votes.
    # The motivation is to use confidence levels as a way to break ties in
    # the votes without switching any decision made based on a difference
    # of 1 vote.
    eps = np.finfo(sum_of_confidences.dtype).eps
    max_abs_confidence = max(abs(max_confidences), abs(min_confidences))
    scale = (0.5 - eps) / max_abs_confidence
    return votes + sum_of_confidences * scale


class LibSVM_SVC(AutoSklearnClassificationAlgorithm):
    def __init__(self, C, kernel, gamma, shrinking, tol, max_iter,
                 class_weight=None, degree=3, coef0=0, random_state=None):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        import sklearn.svm

        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            if soft > 0:
                soft /= 1024 * 1024
                maxrss = resource.getrusage(resource.RUSAGE_SELF)[2] / 1024
                cache_size = (soft - maxrss) / 1.5
            else:
                cache_size = 200
        except Exception:
            cache_size = 200

        self.C = float(self.C)
        if self.degree is None:
            self.degree = 3
        else:
            self.degree = int(self.degree)
        if self.gamma is None:
            self.gamma = 0.0
        else:
            self.gamma = float(self.gamma)
        if self.coef0 is None:
            self.coef0 = 0.0
        else:
            self.coef0 = float(self.coef0)
        self.tol = float(self.tol)
        self.max_iter = float(self.max_iter)
        self.shrinking = self.shrinking == 'True'

        if self.class_weight == "None":
            self.class_weight = None

        self.estimator = sklearn.svm.SVC(C=self.C,
                                         kernel=self.kernel,
                                         degree=self.degree,
                                         gamma=self.gamma,
                                         coef0=self.coef0,
                                         shrinking=self.shrinking,
                                         tol=self.tol,
                                         class_weight=self.class_weight,
                                         max_iter=self.max_iter,
                                         random_state=self.random_state,
                                         cache_size=cache_size)
                                         # probability=True)
        self.estimator.fit(X, Y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        # return self.estimator.predict_proba(X)
        decision = self.estimator.decision_function(X)
        if len(self.estimator.classes_) > 2:
            decision = _ovr_decision_function(decision < 0, decision,
                                              len(self.estimator.classes_))
        return softmax(decision)


    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LibSVM-SVC',
            'name': 'LibSVM Support Vector Classification',
            'handles_regression': False,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': False,
            'is_deterministic': True,
            'input': (DENSE, SPARSE, UNSIGNED_DATA),
            'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True,
                                       default=1.0)
        # No linear kernel here, because we have liblinear
        kernel = CategoricalHyperparameter(name="kernel",
                                           choices=["rbf", "poly", "sigmoid"],
                                           default="rbf")
        degree = UniformIntegerHyperparameter("degree", 1, 5, default=3)
        gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8,
                                           log=True, default=0.1)
        # TODO this is totally ad-hoc
        coef0 = UniformFloatHyperparameter("coef0", -1, 1, default=0)
        # probability is no hyperparameter, but an argument to the SVM algo
        shrinking = CategoricalHyperparameter("shrinking", ["True", "False"],
                                              default="True")
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default=1e-4,
                                         log=True)
        # cache size is not a hyperparameter, but an argument to the program!
        max_iter = UnParametrizedHyperparameter("max_iter", -1)

        cs = ConfigurationSpace()
        cs.add_hyperparameter(C)
        cs.add_hyperparameter(kernel)
        cs.add_hyperparameter(degree)
        cs.add_hyperparameter(gamma)
        cs.add_hyperparameter(coef0)
        cs.add_hyperparameter(shrinking)
        cs.add_hyperparameter(tol)
        cs.add_hyperparameter(max_iter)

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        cs.add_condition(degree_depends_on_poly)
        cs.add_condition(coef0_condition)

        return cs
