import sklearn.naive_bayes

from autosklearn.pipeline.components.classification.bernoulli_nb import \
    BernoulliNB

from .test_base import BaseClassificationComponentTest


class BernoulliNBComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.26
    res["default_iris_iterative"] = 0.26
    res["default_iris_proba"] = 1.1157508543538652
    res["default_iris_sparse"] = 0.38
    res["default_digits"] = 0.81238615664845171
    res["default_digits_iterative"] = 0.81238615664845171
    res["default_digits_binary"] = 0.99392835458409234
    res["default_digits_multilabel"] = 0.73112394623587451
    res["default_digits_multilabel_proba"] = 0.66666666666666663

    sk_mod = sklearn.naive_bayes.BernoulliNB
    module = BernoulliNB