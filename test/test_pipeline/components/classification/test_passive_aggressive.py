import sklearn.linear_model

from autosklearn.pipeline.components.classification.passive_aggressive import \
    PassiveAggressive

from .test_base import BaseClassificationComponentTest


class PassiveAggressiveComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.98
    res["iris_n_calls"] = 6
    res["default_iris_iterative"] = res['default_iris']
    res["iris_iterative_n_iter"] = 64
    res["default_iris_proba"] = 0.27840521921952033
    res["default_iris_sparse"] = 0.48
    res["default_digits"] = 0.9162112932604736
    res["digits_n_calls"] = 6
    res["default_digits_iterative"] = res['default_digits']
    res["digits_iterative_n_iter"] = 64
    res["default_digits_binary"] = 0.99210686095932
    res["default_digits_multilabel"] = 0.910908768565592
    res["default_digits_multilabel_proba"] = 1.0

    sk_mod = sklearn.linear_model.PassiveAggressiveClassifier
    module = PassiveAggressive

    step_hyperparameter = {
        'name': 'max_iter',
        'value': module.get_max_iter(),
    }
