import sklearn.naive_bayes

from autosklearn.pipeline.components.classification.gaussian_nb import \
    GaussianNB

from .test_base import BaseClassificationComponentTest


class GaussianNBComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.95999999999999996
    res["iris_n_calls"] = 1
    res["default_iris_iterative"] = 0.95999999999999996
    res["default_iris_proba"] = 0.11199001987342033
    res["default_iris_sparse"] = -1
    res["default_digits"] = 0.80692167577413476
    res["digits_n_calls"] = 1
    res["default_digits_iterative"] = 0.80692167577413476
    res["default_digits_binary"] = 0.98664238008500305
    res["default_digits_multilabel"] = 0.54135471896765841
    res["default_digits_multilabel_proba"] = 0.99028976450984096

    sk_mod = sklearn.naive_bayes.GaussianNB
    module = GaussianNB
