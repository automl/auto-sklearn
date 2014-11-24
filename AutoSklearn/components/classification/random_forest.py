import sklearn.ensemble

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter

from ..classification_base import AutoSklearnClassificationAlgorithm

class RandomForest(AutoSklearnClassificationAlgorithm):
    def __init__(self, n_estimators, criterion, max_features,
                 max_depth, min_samples_split, min_samples_leaf,
                 bootstrap, random_state=None, n_jobs=1):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        self.n_estimators = int(self.n_estimators)
        if self.max_depth == "Non_":
            self.max_depth = None
        elif self.max_depth is not None:
            self.max_depth = int(self.max_depth)
        self.min_samples_split = int(self.min_samples_split)
        self.min_samples_leaf = int(self.min_samples_leaf)
        if self.max_features not in ("sqrt", "log2", "auto"):
            self.max_features = float(self.max_features)
        if self.bootstrap == "True":
            self.bootstrap = True
        else:
            self.bootstrap = False

        self.estimator = sklearn.ensemble.RandomForestClassifier(
            n_estimators=self.n_estimators, criterion=self.criterion,
            max_depth=self.max_depth, min_samples_split=self
            .min_samples_split, min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features, random_state=self.random_state,
            n_jobs=self.n_jobs)
        self.estimator.fit(X, Y)

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def handles_missing_values(self):
        return False

    def handles_nominal_features(self):
        return False

    def handles_numeric_features(self):
        return True

    def handles_non_binary_classes(self):
        # TODO: describe whether by OneVsOne or OneVsTheRest
        return True

    @staticmethod
    def get_meta_information():
        return {'shortname': 'RF',
                'name': 'Random Forest'}

    @staticmethod
    def get_hyperparameter_search_space():
        n_estimators = UniformIntegerHyperparameter(
            "n_estimators", 10, 100, default=10)
        criterion = CategoricalHyperparameter(
            "criterion", ["gini", "entropy"], default="gini")
        max_features = UniformFloatHyperparameter(
            "max_features", 0.01, 1.0, default=1.0)
        # Don't know how to parametrize this...RF should rather be
        # regularized by the other parameters
        # max_depth = hp_uniform("max_depth", lower, upper)
        max_depth = UnParametrizedHyperparameter("max_depth", "None")
        min_samples_split = UniformIntegerHyperparameter(
            "min_samples_split", 1, 20, default=2)
        min_samples_leaf = UniformIntegerHyperparameter(
            "min_samples_leaf", 1, 20, default=1)
        bootstrap = CategoricalHyperparameter(
            "bootstrap", ["True", "False"], default="True")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(n_estimators)
        cs.add_hyperparameter(criterion)
        cs.add_hyperparameter(max_features)
        cs.add_hyperparameter(max_depth)
        cs.add_hyperparameter(min_samples_split)
        cs.add_hyperparameter(min_samples_leaf)
        cs.add_hyperparameter(bootstrap)
        return cs

    @staticmethod
    def get_all_accepted_hyperparameter_names():
        return (["n_estimators", "criterion", "max_features",
                 "min_samples_split", "min_samples_leaf", "bootstrap"])

    def __str__(self):
        return "AutoSklearn LibSVM Classifier"
