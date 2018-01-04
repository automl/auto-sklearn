import sklearn.neighbors

from autosklearn.pipeline.components.classification.k_nearest_neighbors import \
    KNearestNeighborsClassifier

from .test_base import BaseClassificationComponentTest


class KNearestNeighborsComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.959999999999999
    res["default_iris_iterative"] = -1
    res["default_iris_proba"] = 1.381551055796429
    res["default_iris_sparse"] = 0.82
    res["default_digits"] = 0.93321190042501523
    res["default_digits_iterative"] = -1
    res["default_digits_binary"] = 0.99574984820886459
    res["default_digits_multilabel"] = 0.93433756191199024
    res["default_digits_multilabel_proba"] = 0.9713841334968244

    sk_mod = sklearn.neighbors.KNeighborsClassifier
    module = KNearestNeighborsClassifier