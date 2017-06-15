import sklearn.discriminant_analysis

from autosklearn.pipeline.components.classification.qda import QDA

from .test_base import BaseClassificationComponentTest


class QDAComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 1
    res["default_iris_iterative"] = -1
    res["default_iris_proba"] = 0.56121944069862362
    res["default_iris_sparse"] = -1
    res["default_digits"] = 0.18882817243472982
    res["default_digits_iterative"] = -1
    res["default_digits_binary"] = 0.89071038251366119
    res["default_digits_multilabel"] = 0.17238069061487776
    res["default_digits_multilabel_proba"] = 1.0

    sk_mod = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
    module = QDA
