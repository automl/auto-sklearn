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
    res["default_digits"] = 0.88585306618093507
    res["default_digits_iterative"] = 0.78870673952641168
    res["default_digits_binary"] = 0.98664238008500305
    res["default_digits_multilabel"] = 0.79662039405484386
    res["default_digits_multilabel_proba"] = 0.99252721833266977

    sk_mod = sklearn.ensemble.RandomForestClassifier
    module = RandomForest
