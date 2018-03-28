import sklearn.ensemble

from autosklearn.pipeline.components.classification.random_forest import \
    RandomForest

from .test_base import BaseClassificationComponentTest


class RandomForestComponentTest(BaseClassificationComponentTest):

    __test__ = True

    res = dict()
    res["default_iris"] = 0.94
    res["iris_n_calls"] = 7
    res["default_iris_iterative"] = 0.95999999999999996
    res["default_iris_proba"] = 0.1357018315377785
    res["default_iris_sparse"] = 0.85999999999999999
    # TODO find source of discrepancy!
    res["default_digits"] = 0.899210686095932
    res["digits_n_calls"] = 7
    res["default_digits_iterative"] = 0.8828172434729812
    res["default_digits_binary"] = 0.9884638737097754
    res["default_digits_multilabel"] = 0.9965409490589346
    res["default_digits_multilabel_proba"] = 0.9922896819782431

    sk_mod = sklearn.ensemble.RandomForestClassifier
    module = RandomForest
    step_hyperparameter = {
        'name': 'n_estimators',
        'value': 100,
    }

