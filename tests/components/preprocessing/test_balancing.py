__author__ = 'feurerm'

import unittest

import numpy as np
import sklearn.metrics

from HPOlibConfigSpace.hyperparameters import InactiveHyperparameter

from ParamSklearn.components.preprocessing.balancing import Balancing
from ParamSklearn.classification import ParamSklearnClassifier
from ParamSklearn.components.classification.adaboost import AdaboostClassifier
from ParamSklearn.components.classification.decision_tree import DecisionTree
from ParamSklearn.components.classification.extra_trees import ExtraTreesClassifier
from ParamSklearn.components.classification.gradient_boosting import GradientBoostingClassifier
from ParamSklearn.components.classification.random_forest import RandomForest
from ParamSklearn.components.classification.liblinear_svc import LibLinear_SVC
from ParamSklearn.components.classification.libsvm_svc import LibSVM_SVC
from ParamSklearn.components.classification.sgd import SGD
from ParamSklearn.components.classification.ridge import Ridge
from ParamSklearn.components.preprocessing\
    .extra_trees_preproc_for_classification import ExtraTreesPreprocessor
from ParamSklearn.components.preprocessing.liblinear_svc_preprocessor import LibLinear_Preprocessor
from ParamSklearn.components.preprocessing.random_trees_embedding import RandomTreesEmbedding
from ParamSklearn.util import get_dataset


class BalancingComponentTest(unittest.TestCase):
    def test_balancing_get_weights_treed_single_label(self):
        Y = np.array([0] * 80 + [1] * 20)
        balancing = Balancing(strategy='weighting')
        init_params, fit_params = balancing.get_weights(
            Y, 'random_forest', None, None, None)
        self.assertTrue(np.allclose(fit_params['random_forest:sample_weight'],
                                    np.array([0.4] * 80 + [1.6] * 20)))
        init_params, fit_params = balancing.get_weights(
            Y, None, 'extra_trees_preproc_for_classification', None, None)
        self.assertTrue(np.allclose(fit_params['extra_trees_preproc_for_classification:sample_weight'],
                                    np.array([0.4] * 80 + [1.6] * 20)))

    def test_balancing_get_weights_treed_multilabel(self):
        Y = np.array([[0, 0, 0]] * 100 + [[1, 0, 0]] * 100 + [[0, 1, 0]] * 100 +
                     [[1, 1, 0]] * 100 + [[0, 0, 1]] * 100 + [[1, 0, 1]] * 10)
        balancing = Balancing(strategy='weighting')
        init_params, fit_params = balancing.get_weights(
            Y, 'random_forest', None, None, None)
        self.assertTrue(np.allclose(fit_params['random_forest:sample_weight'],
                                    np.array([0.4] * 500 + [4.0] * 10)))
        init_params, fit_params = balancing.get_weights(
            Y, None, 'extra_trees_preproc_for_classification', None, None)
        self.assertTrue(np.allclose(fit_params['extra_trees_preproc_for_classification:sample_weight'],
                                    np.array([0.4] * 500 + [4.0] * 10)))

    def test_balancing_get_weights_svm_sgd(self):
        Y = np.array([0] * 80 + [1] * 20)
        balancing = Balancing(strategy='weighting')
        init_params, fit_params = balancing.get_weights(
            Y, 'libsvm_svc', None, None, None)
        self.assertEqual(("libsvm_svc:class_weight", "auto"),
                         init_params.items()[0])
        init_params, fit_params = balancing.get_weights(
            Y, None, 'liblinear_svc_preprocessor', None, None)
        self.assertEqual(("liblinear_svc_preprocessor:class_weight", "auto"),
                         init_params.items()[0])

    def test_balancing_get_weights_ridge(self):
        Y = np.array([0] * 80 + [1] * 20)
        balancing = Balancing(strategy='weighting')
        init_params, fit_params = balancing.get_weights(
            Y, 'ridge', None, None, None)
        self.assertAlmostEqual(0.4, init_params['ridge:class_weight'][0])
        self.assertAlmostEqual(1.6, init_params['ridge:class_weight'][1])

    def test_weighting_effect(self):
        for name, clf, acc_no_weighting, acc_weighting in \
                [('adaboost', AdaboostClassifier, 0.692, 0.719),
                 ('decision_tree', DecisionTree, 0.712, 0.668),
                 ('extra_trees', ExtraTreesClassifier, 0.910, 0.913),
                 ('random_forest', RandomForest, 0.896, 0.895),
                 ('libsvm_svc', LibSVM_SVC, 0.915, 0.937),
                 ('liblinear_svc', LibLinear_SVC, 0.920, 0.923),
                 ('sgd', SGD, 0.879, 0.906),
                 ('ridge', Ridge, 0.868, 0.880)]:
            for strategy, acc in [('none', acc_no_weighting),
                                  ('weighting', acc_weighting)]:
                X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
                cs = ParamSklearnClassifier.get_hyperparameter_search_space(
                    include_estimators=[name])
                default = cs.get_default_configuration()
                default.values['balancing:strategy'].value = strategy
                classifier = ParamSklearnClassifier(default, random_state=1)
                predictor = classifier.fit(X_train, Y_train)
                predictions = predictor.predict(X_test)
                self.assertAlmostEqual(acc,
                    sklearn.metrics.accuracy_score(predictions, Y_test),
                    places=3)

        for name, pre, acc_no_weighting, acc_weighting in \
                [('extra_trees_preproc_for_classification',
                  ExtraTreesPreprocessor, 0.900, 0.908),
                   ('liblinear_svc_preprocessor', LibLinear_Preprocessor,
                    0.907, 0.882)]:
            for strategy, acc in [('none', acc_no_weighting),
                                  ('weighting', acc_weighting)]:
                X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')

                cs = ParamSklearnClassifier.get_hyperparameter_search_space(
                    include_estimators=['sgd'], include_preprocessors=[name])
                default = cs.get_default_configuration()
                default.values['balancing:strategy'].value = strategy
                classifier = ParamSklearnClassifier(default, random_state=1)
                predictor = classifier.fit(X_train, Y_train)
                predictions = predictor.predict(X_test)
                self.assertAlmostEqual(acc,
                                       sklearn.metrics.accuracy_score(
                                           predictions, Y_test),
                                       places=3)