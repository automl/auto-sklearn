import sklearn.svm

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant, UnParametrizedHyperparameter
from HPOlibConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction

from ..classification_base import AutoSklearnClassificationAlgorithm

class LibLinear_SVC(AutoSklearnClassificationAlgorithm):
    # Liblinear is not deterministic as it uses a RNG inside
    # TODO: maybe add dual and crammer-singer?
    def __init__(self, penalty, loss, dual, tol, C, class_weight,
            random_state=None):
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, Y):
        self.C = float(self.C)
        self.tol = float(self.tol)

        if self.dual == "False":
            self.dual = False
        elif self.dual == "True":
            self.dual = True
        else:
            raise ValueError("Parameter dual '%s' not in ['True', 'False']" %
                             (self.dual))

        if self.class_weight == "None":
            self.class_weight = None

        self.estimator = sklearn.svm.LinearSVC(penalty=self.penalty,
                                               loss=self.loss,
                                               dual=self.dual,
                                               tol=self.tol,
                                               C=self.C,
                                               class_weight=self.class_weight,
                                               random_state=self.random_state)
        return self.estimator.fit(X, Y)

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def scores(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.decision_function(X)

    @staticmethod
    def get_meta_information():
        return {'shortname': 'Liblinear-SVC',
                'name': 'Liblinear Support Vector Classification',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                # Find out if this is good because of sparsity
                'prefers_data_normalized': False,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': False,
                # TODO find out of this is right!
                # this here suggests so http://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
                'handles_sparse': True,
                # TODO find out what is best used here!
                'preferred_dtype' : None}

    @staticmethod
    def get_hyperparameter_search_space():
        penalty = CategoricalHyperparameter("penalty", ["l1", "l2"],
                                            default="l2")
        loss = CategoricalHyperparameter("loss", ["l1", "l2"], default="l2")
        dual = Constant("dual", "False")
        # This is set ad-how
        tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default=1e-4,
                                         log=True)
        C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True,
                                       default=1.0)
        multi_class = UnParametrizedHyperparameter("multi_class", "ovr")
        # These are set ad-hoc
        fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")
        intercept_scaling = UnParametrizedHyperparameter("intercept_scaling", 1)
        # This does not allow for other resampling methods!
        class_weight = CategoricalHyperparameter("class_weight",
                                                 ["None", "auto"],
                                                 default="None")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(penalty)
        cs.add_hyperparameter(loss)
        cs.add_hyperparameter(dual)
        cs.add_hyperparameter(tol)
        cs.add_hyperparameter(C)
        cs.add_hyperparameter(multi_class)
        cs.add_hyperparameter(fit_intercept)
        cs.add_hyperparameter(intercept_scaling)
        cs.add_hyperparameter(class_weight)
        penalty_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(penalty, "l1"),
            ForbiddenEqualsClause(loss, "l1")
        )
        constant_penalty_and_loss = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, "False"),
            ForbiddenEqualsClause(penalty, "l2"),
            ForbiddenEqualsClause(loss, "l1")
        )
        cs.add_forbidden_clause(penalty_and_loss)
        cs.add_forbidden_clause(constant_penalty_and_loss)
        return cs

    def __str__(self):
        return "AutoSklearn Liblinear Classifier"
