import sklearn.ensemble

from autosklearn.pipeline.components.classification.random_forest import \
    RandomForest

from .test_base import BaseClassificationComponentTest


class RandomForestComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.96
    res["iris_n_calls"] = 9
    res["default_iris_iterative"] = res['default_iris']
    res["default_iris_proba"] = 0.0996785324703419
    res["default_iris_sparse"] = 0.85999999999999999
    res["default_digits"] = 0.8998178506375227
    res["digits_n_calls"] = 9
    res["default_digits_iterative"] = res['default_digits']
    res["default_digits_binary"] = 0.9896782027929569
    res["default_digits_multilabel"] = 0.9973653110879388
    res["default_digits_multilabel_proba"] = 0.9965660960196189

    sk_mod = sklearn.ensemble.RandomForestClassifier
    module = RandomForest
    step_hyperparameter = {
        'name': 'n_estimators',
        'value': module.get_max_iter(),
    }
