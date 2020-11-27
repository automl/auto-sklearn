import sklearn.linear_model

from autosklearn.pipeline.components.classification.passive_aggressive import \
    PassiveAggressive

from .test_base import BaseClassificationComponentTest


class PassiveAggressiveComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.92
    res["iris_n_calls"] = 5
    res["default_iris_iterative"] = 0.92
    res["iris_iterative_n_iter"] = 32
    res["default_iris_proba"] = 0.29271032477461295
    res["default_iris_sparse"] = 0.4
    res["default_digits"] = 0.9156041287188829
    res["digits_n_calls"] = 6
    res["default_digits_iterative"] = 0.9156041287188829
    res["digits_iterative_n_iter"] = 64
    res["default_digits_binary"] = 0.9927140255009107
    res["default_digits_multilabel"] = 0.90997912489192
    res["default_digits_multilabel_proba"] = 1.0
    res['ignore_hps'] = ['max_iter']

    sk_mod = sklearn.linear_model.PassiveAggressiveClassifier
    module = PassiveAggressive

    step_hyperparameter = {
        'name': 'max_iter',
        'value': module.get_max_iter(),
    }
