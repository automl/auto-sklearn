import sklearn.linear_model

from autosklearn.pipeline.components.classification.passive_aggressive import \
    PassiveAggressive

from .test_base import BaseClassificationComponentTest


class PassiveAggressiveComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.66000000000000003
    res["default_iris_iterative"] = 0.95999999999999996
    res["default_iris_proba"] = 0.52834934001514167
    res["default_iris_sparse"] = 0.38
    res["default_digits"] = 0.91803278688524592
    res["default_digits_iterative"] = 0.9174256223436551
    res["default_digits_binary"] = 0.99089253187613846
    res["default_digits_multilabel"] = 0.93818765184149111
    res["default_digits_multilabel_proba"] = 1.0
    res['ignore_hps'] = ['max_iter']

    sk_mod = sklearn.linear_model.PassiveAggressiveClassifier
    module = PassiveAggressive
