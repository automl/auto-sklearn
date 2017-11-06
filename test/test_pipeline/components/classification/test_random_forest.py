import sklearn.ensemble

from autosklearn.pipeline.components.classification.random_forest import \
    RandomForest

from .test_base import BaseClassificationComponentTest


class RandomForestComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.95999999999999996
    res["default_iris_iterative"] = 0.95999999999999996
    res["default_iris_proba"] = 0.12735301219200268
    res["default_iris_sparse"] = 0.85999999999999999
    res["default_digits"] = 0.88281724347298118
    res["default_digits_iterative"] = 0.77231329690346084
    res["default_digits_binary"] = 0.98907103825136611
    res["default_digits_multilabel"] = 0.99232171389942414
    res["default_digits_multilabel_proba"] = 0.99514488225492048

    sk_mod = sklearn.ensemble.RandomForestClassifier
    module = RandomForest
