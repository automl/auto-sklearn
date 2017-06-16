import sklearn.discriminant_analysis

from autosklearn.pipeline.components.classification.qda import QDA

from .test_base import BaseClassificationComponentTest


class QDAComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 1.0
    res["default_iris_iterative"] = -1
    res["default_iris_proba"] = 0.56124476634783993
    res["default_iris_sparse"] = -1
    res["default_digits"] = 0.18882817243472982
    res["default_digits_iterative"] = -1
    res["default_digits_binary"] = 0.89071038251366119
    res["default_digits_multilabel"] = 0.17011293429111091
    res["default_digits_multilabel_places"] = 1
    res["default_digits_multilabel_proba"] = 0.99999999999999989

    sk_mod = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
    module = QDA
