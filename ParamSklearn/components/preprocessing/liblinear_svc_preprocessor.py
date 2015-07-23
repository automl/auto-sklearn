import sklearn.svm

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant, UnParametrizedHyperparameter
from HPOlibConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction

from ParamSklearn.components.base import \
    ParamSklearnPreprocessingAlgorithm
from ParamSklearn.util import SPARSE, DENSE, INPUT


class LibLinear_Preprocessor(ParamSklearnPreprocessingAlgorithm):
    def __init__(self, penalty, loss, dual, tol, C, multi_class,
                 fit_intercept, intercept_scaling, class_weight,
                 random_state=None):
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.preprocessor = None

    def fit(self, X, Y):
        self.C = float(self.C)
        self.tol = float(self.tol)

        self.dual = self.dual == 'True'
        self.fit_intercept = self.fit_intercept == 'True'
        self.intercept_scaling = float(self.intercept_scaling)

        if self.class_weight == "None":
            self.class_weight = None

        self.preprocessor = sklearn.svm.LinearSVC(penalty=self.penalty,
                                               loss=self.loss,
                                               dual=self.dual,
                                               tol=self.tol,
                                               C=self.C,
                                               class_weight=self.class_weight,
                                               random_state=self.random_state)
        self.preprocessor.fit(X, Y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties():
        return {'shortname': 'Liblinear-Preprocessor',
                'name': 'Liblinear Support Vector Preprocessing',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': True,
                # Find out if this is good because of sparsity
                'prefers_data_normalized': False,
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': False,
                # TODO find out of this is right!
                # this here suggests so http://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use
                'handles_sparse': True,
                'input': (SPARSE, DENSE),
                'output': INPUT,
                # TODO find out what is best used here!
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
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
        penalty_and_dual = ForbiddenAndConjunction(
            ForbiddenEqualsClause(dual, "False"),
            ForbiddenEqualsClause(penalty, "l1")
        )
        cs.add_forbidden_clause(penalty_and_loss)
        cs.add_forbidden_clause(constant_penalty_and_loss)
        cs.add_forbidden_clause(penalty_and_dual)
        return cs
