import sklearn.ensemble

from autosklearn.pipeline.components.classification.gradient_boosting import \
    GradientBoostingClassifier

from .test_base import BaseClassificationComponentTest


class GradientBoostingComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.92
    res["iris_n_calls"] = 7
    res["default_iris_iterative"] = 0.92
    res["default_iris_proba"] = 0.48109031836615801
    res["default_iris_sparse"] = -1
    res["default_digits"] = 0.80206435944140864
    res["digits_n_calls"] = 7
    res["default_digits_iterative"] = 0.80206435944140864
    res["default_digits_binary"] = 0.98178506375227692
    res["default_digits_multilabel"] = -1
    res["default_digits_multilabel_proba"] = -1

    sk_mod = sklearn.ensemble.ExtraTreesClassifier
    module = GradientBoostingClassifier
    step_hyperparameter = {
        'name': 'n_estimators',
        'value': 100,
    }