import sklearn.ensemble

from autosklearn.pipeline.components.classification.random_forest import \
    RandomForest

from .test_base import BaseClassificationComponentTest


class RandomForestComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.96
    res["iris_n_calls"] = 9
    res["default_iris_iterative"] = 0.95999999999999996
    res["default_iris_proba"] = 0.09922562116542713
    res["default_iris_sparse"] = 0.85999999999999999
    # TODO find source of discrepancy!
    res["default_digits"] = 0.9058894960534305
    res["digits_n_calls"] = 9
    res["default_digits_iterative"] = 0.9058894960534305
    res["default_digits_binary"] = 0.9914996964177292
    res["default_digits_multilabel"] = 0.9957676902536715
    res["default_digits_multilabel_proba"] = 0.9965660960196189

    sk_mod = sklearn.ensemble.RandomForestClassifier
    module = RandomForest
    step_hyperparameter = {
        'name': 'n_estimators',
        'value': module.get_max_iter(),
    }
