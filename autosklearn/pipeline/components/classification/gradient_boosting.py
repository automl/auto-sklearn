import numpy as np

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant, \
    CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition

from autosklearn.pipeline.components.base import (
    AutoSklearnClassificationAlgorithm,
    IterativeComponent,
)
from autosklearn.pipeline.constants import DENSE, UNSIGNED_DATA, PREDICTIONS
from autosklearn.util.common import check_none


class GradientBoostingClassifier(
    IterativeComponent,
    AutoSklearnClassificationAlgorithm
):
    def __init__(self, loss, learning_rate, min_samples_leaf, max_depth,
                 max_leaf_nodes, max_bins, l2_regularization, early_stop, tol, scoring,
                 n_iter_no_change=0, validation_fraction=None, random_state=None,
                 verbose=0):
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = self.get_max_iter()
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.max_bins = max_bins
        self.l2_regularization = l2_regularization
        self.early_stop = early_stop
        self.tol = tol
        self.scoring = scoring
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose
        self.estimator = None
        self.fully_fit_ = False

    @staticmethod
    def get_max_iter():
        return 512

    def get_current_iter(self):
        return self.estimator.n_iter_

    def iterative_fit(self, X, y, n_iter=2, refit=False):

        """
        Set n_iter=2 for the same reason as for SGD
        """
        import sklearn.ensemble
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa

        if refit:
            self.estimator = None

        if self.estimator is None:
            self.fully_fit_ = False

            self.learning_rate = float(self.learning_rate)
            self.max_iter = int(self.max_iter)
            self.min_samples_leaf = int(self.min_samples_leaf)
            if check_none(self.max_depth):
                self.max_depth = None
            else:
                self.max_depth = int(self.max_depth)
            if check_none(self.max_leaf_nodes):
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(self.max_leaf_nodes)
            self.max_bins = int(self.max_bins)
            self.l2_regularization = float(self.l2_regularization)
            self.tol = float(self.tol)
            if check_none(self.scoring):
                self.scoring = None
            if self.early_stop == "off":
                self.n_iter_no_change = 0
                self.validation_fraction_ = None
            elif self.early_stop == "train":
                self.n_iter_no_change = int(self.n_iter_no_change)
                self.validation_fraction_ = None
            elif self.early_stop == "valid":
                self.n_iter_no_change = int(self.n_iter_no_change)
                self.validation_fraction = float(self.validation_fraction)
                n_classes = len(np.unique(y))
                if self.validation_fraction * X.shape[0] < n_classes:
                    self.validation_fraction_ = n_classes
                else:
                    self.validation_fraction_ = self.validation_fraction
            else:
                raise ValueError("early_stop should be either off, train or valid")
            self.verbose = int(self.verbose)
            n_iter = int(np.ceil(n_iter))

            # initial fit of only increment trees
            self.estimator = sklearn.ensemble.HistGradientBoostingClassifier(
                loss=self.loss,
                learning_rate=self.learning_rate,
                max_iter=n_iter,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                max_leaf_nodes=self.max_leaf_nodes,
                max_bins=self.max_bins,
                l2_regularization=self.l2_regularization,
                tol=self.tol,
                scoring=self.scoring,
                n_iter_no_change=self.n_iter_no_change,
                validation_fraction=self.validation_fraction_,
                verbose=self.verbose,
                warm_start=True,
                random_state=self.random_state,
            )
        else:
            self.estimator.max_iter += n_iter
            self.estimator.max_iter = min(self.estimator.max_iter,
                                          self.max_iter)

        self.estimator.fit(X, y)

        if self.estimator.max_iter >= self.max_iter \
           or self.estimator.max_iter > self.estimator.n_iter_:

            self.fully_fit_ = True

        return self

    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        elif not hasattr(self, 'fully_fit_'):
            return False
        else:
            return self.fully_fit_

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GB',
                'name': 'Gradient Boosting Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'handles_multioutput': False,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        loss = Constant("loss", "auto")
        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=200, default_value=20, log=True)
        max_depth = UnParametrizedHyperparameter(
            name="max_depth", value="None")
        max_leaf_nodes = UniformIntegerHyperparameter(
            name="max_leaf_nodes", lower=3, upper=2047, default_value=31, log=True)
        max_bins = Constant("max_bins", 255)
        l2_regularization = UniformFloatHyperparameter(
            name="l2_regularization", lower=1E-10, upper=1, default_value=1E-10, log=True)
        # Train temporarily disabled to avoid error with scikit-learn 0.22
        # TODO re-enable with later scikit-learn version
        early_stop = CategoricalHyperparameter(
            name="early_stop", choices=["off", "valid"], default_value="off")
        tol = UnParametrizedHyperparameter(
            name="tol", value=1e-7)
        scoring = UnParametrizedHyperparameter(
            name="scoring", value="loss")
        n_iter_no_change = UniformIntegerHyperparameter(
            name="n_iter_no_change", lower=1, upper=20, default_value=10)
        validation_fraction = UniformFloatHyperparameter(
            name="validation_fraction", lower=0.01, upper=0.4, default_value=0.1)

        cs.add_hyperparameters([loss, learning_rate, min_samples_leaf,
                                max_depth, max_leaf_nodes, max_bins, l2_regularization,
                                early_stop, tol, scoring, n_iter_no_change,
                                validation_fraction])

        n_iter_no_change_cond = InCondition(
            n_iter_no_change, early_stop, ["valid"])  # , "train"])
        validation_fraction_cond = EqualsCondition(
            validation_fraction, early_stop, "valid")

        cs.add_conditions([n_iter_no_change_cond, validation_fraction_cond])

        return cs
