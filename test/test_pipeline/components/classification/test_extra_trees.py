import sklearn.ensemble

from autosklearn.pipeline.components.classification.extra_trees import \
    ExtraTreesClassifier

from .test_base import BaseClassificationComponentTest


class ExtraTreesComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.95999999999999996
    res["default_iris_iterative"] = 0.93999999999999995
    res["default_iris_proba"] = 0.1086791056721286
    res["default_iris_sparse"] = 0.71999999999999997
    res["default_digits"] = 0.8986035215543412
    res["default_digits_iterative"] = 0.81785063752276865
    res["default_digits_binary"] = 0.99392835458409234
    res["default_digits_multilabel"] = 0.82229416703109792
    res["default_digits_multilabel_proba"] = 0.99401797442008899

    sk_mod = sklearn.ensemble.ExtraTreesClassifier
    module = ExtraTreesClassifier