import sklearn.svm

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant
from HPOlibConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction

from ..classification_base import AutoSklearnClassificationAlgorithm

class LibLinear_SVC(AutoSklearnClassificationAlgorithm):
    # Liblinear is not deterministic as it uses a RNG inside
    # TODO: maybe add dual and crammer-singer?
    def __init__(self, penalty, loss, C, dual, random_state=None,):
        self.penalty = penalty
        self.loss = loss
        self.C = C
        self.dual = dual
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        #if self.LOG2_C is not None:
        #    self.LOG2_C = float(self.LOG2_C)
        #    self.C = 2 ** self.LOG2_C

        if self.dual == "__False__":
            self.dual = False
        elif self.dual == "__True__":
            self.dual = True

        self.C = float(self.C)
        self.estimator = sklearn.svm.LinearSVC(penalty=self.penalty,
                                               loss=self.loss, C=self.C,
                                               random_state=self.random_state,
                                               dual=False)
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
    def get_meta_information():
        return {'shortname': 'Liblinear-SVC',
                'name': 'Liblinear Support Vector Classification'}

    @staticmethod
    def get_hyperparameter_search_space():
        penalty = CategoricalHyperparameter("penalty", ["l1", "l2"])
        loss = CategoricalHyperparameter("loss", ["l1", "l2"])
        C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True)
        dual = Constant("dual", "__False__")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(penalty)
        cs.add_hyperparameter(loss)
        cs.add_hyperparameter(C)
        cs.add_hyperparameter(dual)
        penalty_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(penalty, "l1"),
            ForbiddenEqualsClause(loss, "l1")
        )
        constant_penalty_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, "__False__"),
            ForbiddenEqualsClause(penalty, "l2"),
            ForbiddenEqualsClause(loss, "l1")
        )
        cs.add_forbidden_clause(penalty_and_loss)
        cs.add_forbidden_clause(constant_penalty_and_loss)
        return cs

    @staticmethod
    def get_all_accepted_hyperparameter_names():
        return (["LOG2_C", "C", "penalty", "loss"])

    def __str__(self):
        return "AutoSklearn Liblinear Classifier"
