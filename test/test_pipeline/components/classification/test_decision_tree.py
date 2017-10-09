import sklearn.tree

from autosklearn.pipeline.components.classification.decision_tree import \
    DecisionTree

from .test_base import BaseClassificationComponentTest


class DecisionTreetComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.62
    res["default_iris_iterative"] = -1
    res["default_iris_proba"] = 0.51333963481747835
    res["default_iris_sparse"] = 0.41999999999999998
    res["default_digits"] = 0.15057680631451123
    res["default_digits_iterative"] = -1
    res["default_digits_binary"] = 0.92167577413479052
    res["default_digits_multilabel"] = 0.56326230155706081
    res["default_digits_multilabel_proba"] = 0.83333333333333337

    sk_mod = sklearn.tree.DecisionTreeClassifier
    module = DecisionTree