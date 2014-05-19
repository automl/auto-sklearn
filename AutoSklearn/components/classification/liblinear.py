import sklearn.svm

from ...util import hp_uniform, hp_choice
from ..classification_base import AutoSklearnClassificationAlgorithm

class LibLinear_SVC(AutoSklearnClassificationAlgorithm):
     # TODO: maybe add dual and crammer-singer?
    def __init__(self, penalty="l2", loss="l2", C=1.0, LOG2_C=None, random_state=None):
        self.penalty = penalty
        self.loss = loss
        self.C = C
        self.LOG2_C = LOG2_C
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        if self.LOG2_C is not None:
            self.C = 2 ** self.LOG2_C
        self.estimator = sklearn.svm.LinearSVC(penalty=self.penalty,
                                               loss=self.loss, C=self.C,
                                               random_state=self.random_state)
        self.estimator.fit(X, Y)

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
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
        # penalty l1 and loss l1 together are forbidden
        penalty_and_loss = hp_choice("penalty_and_loss",
                                     [{"penalty": "l1", "loss": "l2"},
                                      {"penalty": "l2", "loss": "l1"},
                                      {"penalty": "l2", "loss": "l2"}])
        loss = hp_choice("loss", ["l1", "l2"])
        LOG2_C = hp_uniform("LOG2_C", -5, 15)
        return {"name": "liblinear", "penalty_and_loss": penalty_and_loss,
                "LOG2_C": LOG2_C}

    def __str__(self):
        return "AutoSklearn Liblinear Classifier"
