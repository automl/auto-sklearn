import numpy as np
import sklearn.ensemble
from sklearn.experimental import enable_hist_gradient_boosting


from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant, \
    CategoricalHyperparameter

from autosklearn.pipeline.components.base import (
    AutoSklearnClassificationAlgorithm,
    IterativeComponentWithSampleWeight,
)
from autosklearn.pipeline.constants import *
from autosklearn.util.common import check_none


#class GradientBoostingClassifier(IterativeComponentWithSampleWeight,
#                                 AutoSklearnClassificationAlgorithm):

class GradientBoostingClassifier(AutoSklearnClassificationAlgorithm):    

    #def __init__(self, loss, learning_rate, n_estimators, subsample,
    #             min_samples_split, min_samples_leaf,
    #             min_weight_fraction_leaf, max_depth, criterion, max_features,
    #             max_leaf_nodes, min_impurity_decrease, random_state=None,
    #             verbose=0):

    def __init__(self, learning_rate, min_samples_leaf, max_iter, max_depth, 
                 max_leaf_nodes, random_state=None, verbose=0):

        #self.loss = loss
        self.learning_rate = learning_rate
        #self.n_estimators = n_estimators
        self.max_iter = max_iter
        #self.subsample = subsample
        #self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        #self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        #self.criterion=criterion
        #self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        #self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.verbose = verbose
        self.estimator = None
        self.fully_fit_ = False

    def fit(self, X, Y, sample_weight=None, n_iter=1, refit=False):
        import sklearn.ensemble
        import sklearn.tree

        self.learning_rate = float(self.learning_rate)
        self.max_iter = int(self.max_iter)
        self.min_samples_leaf = int(self.min_samples_leaf)
        if check_none(self.max_leaf_nodes):
            self.max_leaf_nodes = None
        else:
            self.max_leaf_nodes = int(self.max_leaf_nodes)
        if check_none(self.max_depth):
            self.max_depth = None
        else:
            self.max_depth = int(self.max_depth)
        self.verbose = int(self.verbose)

        estimator = sklearn.ensemble.HistGradientBoostingClassifier(
            #loss=self.loss,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            verbose=self.verbose,
            random_state=self.random_state,
            max_iter=n_iter,
            
            #n_estimators=n_iter,
            #subsample=self.subsample,
            #min_samples_split=self.min_samples_split,
            #min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            #criterion=self.criterion,
            #max_features=self.max_features,
        )

        #estimator.fit(X, Y, sample_weight=sample_weight)
        estimator.fit(X, Y)

        self.estimator = estimator
        return self




    def configuration_fully_fitted(self):
        if self.estimator is None:
            return False
        #return not len(self.estimator.estimators_) < self.n_estimators
        return not len(self.estimator.n_iter_) < self.max_iter

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
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        learning_rate = UniformFloatHyperparameter(
            name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
        max_depth = UniformIntegerHyperparameter(
            name="max_depth", lower=2, upper=10, default_value=3)
        min_samples_leaf = UniformIntegerHyperparameter(
            name="min_samples_leaf", lower=1, upper=20, default_value=1)
        max_leaf_nodes = UnParametrizedHyperparameter(
            name="max_leaf_nodes", value="None")
        max_iter = UniformIntegerHyperparameter(
            "max_iter", 50, 500, default_value=100)


        # Commented out
        # Options for HistGradientBoostingClassifier are:
        #   {‘auto’, ‘binary_crossentropy’, ‘categorical_crossentropy’}
        #   default is auto
        #loss = Constant("loss", "deviance") 
        
        #criterion = CategoricalHyperparameter(
        #    'criterion', ['friedman_mse', 'mse', 'mae'],
        #    default_value='mse')
        #min_samples_split = UniformIntegerHyperparameter(
        #    name="min_samples_split", lower=2, upper=20, default_value=2)
        #max_features = UniformFloatHyperparameter(
        #    "max_features", 0.1, 1.0 , default_value=1)
        #min_weight_fraction_leaf = UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
        #subsample = UniformFloatHyperparameter(
        #        name="subsample", lower=0.01, upper=1.0, default_value=1.0)
        #min_impurity_decrease = UnParametrizedHyperparameter(
        #    name='min_impurity_decrease', value=0.0)
        
        #cs.add_hyperparameters([loss, learning_rate, n_estimators, max_depth,
        #                        criterion, min_samples_split, min_samples_leaf,
        #                        min_weight_fraction_leaf, subsample,
        #                        max_features, max_leaf_nodes,
        #                        min_impurity_decrease])

        cs.add_hyperparameters([learning_rate, max_iter, max_depth, 
                                min_samples_leaf, max_leaf_nodes])

        return cs

