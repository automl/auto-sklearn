import sklearn.svm

from autosklearn.pipeline.components.classification.liblinear_svc import \
    LibLinear_SVC

from .test_base import BaseClassificationComponentTest


class LibLinearComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 1
    res["default_iris_iterative"] = -1
    res["default_iris_proba"] = 0.33728319465089696
    res["default_iris_sparse"] = 0.56
    res["default_digits"] = 0.91499696417729204
    res['default_digits_places'] = 2
    res["default_digits_iterative"] = -1
    res["default_digits_binary"] = 0.98907103825136611
    res["default_digits_multilabel"] = 0.89889188078944637
    res["default_digits_multilabel_proba"] = 0.99999999999999989

    sk_mod = sklearn.svm.LinearSVC
    module = LibLinear_SVC