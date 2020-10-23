import sklearn.ensemble

from autosklearn.pipeline.components.classification.gradient_boosting import \
    GradientBoostingClassifier

from .test_base import BaseClassificationComponentTest


class GradientBoostingComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.92
    res["default_iris_iterative"] = 0.92
    res["default_iris_proba"] = 1.1099521844626845
    res["default_iris_sparse"] = -1
    res["default_digits"] = 0.8652094717668488
    res["default_digits_iterative"] = 0.8652094717668488
    res["default_digits_binary"] = 0.9933211900425015
    res["default_digits_multilabel"] = -1
    res["default_digits_multilabel_proba"] = -1

    sk_mod = sklearn.ensemble.ExtraTreesClassifier
    module = GradientBoostingClassifier
    step_hyperparameter = {
        'name': 'max_iter',
        'value': module.get_max_iter(),
    }
