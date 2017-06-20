import sklearn.ensemble

from autosklearn.pipeline.components.classification.gradient_boosting import \
    GradientBoostingClassifier

from .test_base import BaseClassificationComponentTest


class GradientBoostingComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.93999999999999995
    res["default_iris_iterative"] = 0.95999999999999996
    res["default_iris_proba"] = 0.36351844058108812
    res["default_iris_sparse"] = -1
    res["default_digits"] = 0.87795992714025506
    res["default_digits_iterative"] = 0.78324225865209474
    res["default_digits_binary"] = 0.99089253187613846
    res["default_digits_multilabel"] = -1
    res["default_digits_multilabel_proba"] = -1

    sk_mod = sklearn.ensemble.ExtraTreesClassifier
    module = GradientBoostingClassifier