import autosklearn.pipeline.implementations.xgb

from autosklearn.pipeline.components.classification.xgradient_boosting import \
    XGradientBoostingClassifier

from .test_base import BaseClassificationComponentTest


class XGradientBoostingComponentTest(BaseClassificationComponentTest):
    __test__ = True

    res = dict()
    res["default_iris"] = 0.92
    res["iris_n_calls"] = 7
    res["default_iris_iterative"] = 0.92
    res["default_iris_proba"] = 0.22956673529930413
    res["default_iris_sparse"] = 0.88
    res["default_digits"] = 0.8427443837279903
    res["digits_n_calls"] = 7
    res["default_digits_iterative"] = 0.8427443837279903
    res["default_digits_binary"] = 0.9823922282938676
    res["default_digits_multilabel"] = 0.88
    res["default_digits_multilabel_proba"] = 0.88

    sk_mod = autosklearn.pipeline.implementations.xgb.CustomXGBClassifier
    module = XGradientBoostingClassifier
    step_hyperparameter = {
        'name': 'n_estimators',
        'value': 128,
    }