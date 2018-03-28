import sklearn.ensemble

from autosklearn.pipeline.components.classification.extra_trees import \
    ExtraTreesClassifier

from .test_base import BaseClassificationComponentTest


class ExtraTreesComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.96
    res["iris_n_calls"] = 7
    res["default_iris_iterative"] = 0.96
    res["default_iris_proba"] = 0.11193378733757438
    res["default_iris_sparse"] = 0.74
    res["default_digits"] = 0.9137826350941105
    res["digits_n_calls"] = 7
    res["default_digits_iterative"] = 0.914996964177292
    res["default_digits_binary"] = 0.9939283545840923
    res["default_digits_multilabel"] = 1.0
    res["default_digits_multilabel_proba"] = 0.9942832130730052

    sk_mod = sklearn.ensemble.ExtraTreesClassifier
    module = ExtraTreesClassifier
    step_hyperparameter = {
        'name': 'n_estimators',
        'value': 100,
    }