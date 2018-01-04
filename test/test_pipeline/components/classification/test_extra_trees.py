import sklearn.ensemble

from autosklearn.pipeline.components.classification.extra_trees import \
    ExtraTreesClassifier

from .test_base import BaseClassificationComponentTest


class ExtraTreesComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.95999999999999996
    res["default_iris_iterative"] = 0.93999999999999995
    res["default_iris_proba"] = 0.10751929472112302
    res["default_iris_sparse"] = 0.74
    res["default_digits"] = 0.91135397692774744
    res["default_digits_iterative"] = 0.85428051001821492
    res["default_digits_binary"] = 0.99453551912568305
    res["default_digits_multilabel"] = 0.9990291262135923
    res["default_digits_multilabel_proba"] = 0.99628951691504752

    sk_mod = sklearn.ensemble.ExtraTreesClassifier
    module = ExtraTreesClassifier