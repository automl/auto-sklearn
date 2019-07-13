import sklearn.linear_model

from autosklearn.pipeline.components.classification.passive_aggressive import \
    PassiveAggressive

from .test_base import BaseClassificationComponentTest


class PassiveAggressiveComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.78
    res["iris_n_calls"] = 4
    res["default_iris_iterative"] = 0.98
    res["default_iris_proba"] = 0.4471909030527266
    res["default_iris_sparse"] = 0.36
    res["default_digits"] = 0.9156041287188829
    res["digits_n_calls"] = 4
    res["default_digits_iterative"] = 0.9174256223436551
    res["default_digits_binary"] = 0.9927140255009107
    res["default_digits_multilabel"] = 0.93818765184149111
    res["default_digits_multilabel_proba"] = 1.0
    res['ignore_hps'] = ['max_iter']

    sk_mod = sklearn.linear_model.PassiveAggressiveClassifier
    module = PassiveAggressive
