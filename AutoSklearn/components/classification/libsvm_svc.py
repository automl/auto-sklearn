import sklearn.svm

from ...util import hp_uniform
from ..classification_base import AutoSklearnClassificationAlgorithm

class LibSVM_SVC(AutoSklearnClassificationAlgorithm):
    # TODO: maybe ad shrinking to the parameters?
    def __init__(self, C=1.0, gamma=0.0,
                 LOG2_C=None, LOG2_gamma=None,
                 random_state=None):
        self.C = C
        self.gamma = gamma
        self.LOG2_C = LOG2_C
        self.LOG2_gamma = LOG2_gamma
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        if self.LOG2_C is not None:
            self.LOG2_C = float(self.LOG2_C)
            self.C = 2 ** self.LOG2_C
        if self.LOG2_gamma is not None:
            self.LOG2_gamma = float(self.LOG2_gamma)
            self.gamma = 2 ** self.LOG2_gamma

        self.C = float(self.C)
        self.gamma = float(self.gamma)
        self.estimator = sklearn.svm.SVC(C=self.C, gamma=self.gamma,
                                         random_state=self.random_state,
                                         cache_size=2000)
        self.estimator.fit(X, Y)

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    def handles_missing_values(self):
        # TODO: should be able to handle sparse data itself...
        return False

    def handles_nominal_features(self):
        return False

    def handles_numeric_features(self):
        return True

    def handles_non_binary_classes(self):
        # TODO: describe whether by OneVsOne or OneVsTheRest
        return True

    @staticmethod
    def get_hyperparameter_search_space():
        LOG2_C = hp_uniform("LOG2_C", -5, 15)
        LOG2_gamma = hp_uniform("LOG2_gamma", -15, 3)
        return {"name": "libsvm_svc", "LOG2_C": LOG2_C, "LOG2_gamma":
            LOG2_gamma}

    @staticmethod
    def get_all_accepted_hyperparameter_names():
        return (["LOG2_C", "C", "LOG2_gamma", "gamma"])

    def __str__(self):
        return "AutoSklearn LibSVM Classifier"
