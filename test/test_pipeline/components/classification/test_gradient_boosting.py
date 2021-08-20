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

    # NOTE non-determinism, iterative-fit for test_module_idempotent
    #
    #   iterative fit
    #
    #   Firstly, we disable this as calls to `fit` end up calling `iterative_fit`
    #   in the underlying sklearn estimator. The `max_iter` of the underlying
    #   estimator are progressibly updated on each call so `max_iter` will
    #   not be the same on the first fit call as the second fit call.
    #
    #   non-determinism
    #
    #   The non-determinism that cause failures for issue 1209 comes from
    #   sampling configurations from the configuration space itself.
    res["ignore_hps"] = ["max_iter"]

    sk_mod = sklearn.ensemble.ExtraTreesClassifier
    module = GradientBoostingClassifier
    step_hyperparameter = {
        'name': 'max_iter',
        'value': module.get_max_iter(),
    }
