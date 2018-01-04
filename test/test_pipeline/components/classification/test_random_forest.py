import sklearn.ensemble

from autosklearn.pipeline.components.classification.random_forest import \
    RandomForest

from .test_base import BaseClassificationComponentTest


class RandomForestComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.95999999999999996
    res["default_iris_iterative"] = 0.95999999999999996
    res["default_iris_proba"] = 0.12694770574921302
    res["default_iris_sparse"] = 0.85999999999999999
    res["default_digits"] = 0.90771098967820274
    res["default_digits_iterative"] = 0.77231329690346084
    res["default_digits_binary"] = 0.98967820279295693
    res["default_digits_multilabel"] = 0.99421420083184786
    res["default_digits_multilabel_proba"] = 0.9942895994466453

    sk_mod = sklearn.ensemble.RandomForestClassifier
    module = RandomForest
