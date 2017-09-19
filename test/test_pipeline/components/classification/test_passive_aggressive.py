import sklearn.linear_model

from autosklearn.pipeline.components.classification.passive_aggressive import \
    PassiveAggressive

from .test_base import BaseClassificationComponentTest


class PassiveAggressiveComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.71999999999999997
    res["default_iris_iterative"] = 0.81999999999999995
    res["default_iris_proba"] = 0.46713079533751622
    res["default_iris_sparse"] = 0.4
    res["default_digits"] = 0.92046144505160898
    res["default_digits_iterative"] = 0.92349726775956287
    res["default_digits_binary"] = 0.99574984820886459
    res["default_digits_multilabel"] = 0.8975269956947447
    res["default_digits_multilabel_proba"] = 0.99703892466326138

    sk_mod = sklearn.linear_model.PassiveAggressiveClassifier
    module = PassiveAggressive
