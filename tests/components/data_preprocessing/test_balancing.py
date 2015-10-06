__author__ = 'feurerm'

import unittest

import numpy as np
import sklearn.metrics

from ParamSklearn.components.data_preprocessing.balancing import Balancing
from ParamSklearn.classification import ParamSklearnClassifier
from ParamSklearn.components.classification.adaboost import AdaboostClassifier
from ParamSklearn.components.classification.decision_tree import DecisionTree
from ParamSklearn.components.classification.extra_trees import ExtraTreesClassifier
from ParamSklearn.components.classification.gradient_boosting import GradientBoostingClassifier
from ParamSklearn.components.classification.random_forest import RandomForest
from ParamSklearn.components.classification.liblinear_svc import LibLinear_SVC
from ParamSklearn.components.classification.libsvm_svc import LibSVM_SVC
from ParamSklearn.components.classification.sgd import SGD
from ParamSklearn.components.feature_preprocessing\
    .extra_trees_preproc_for_classification import ExtraTreesPreprocessor
from ParamSklearn.components.feature_preprocessing.liblinear_svc_preprocessor import LibLinear_Preprocessor
from ParamSklearn.components.feature_preprocessing.random_trees_embedding import RandomTreesEmbedding
from ParamSklearn.util import get_dataset


class BalancingComponentTest(unittest.TestCase):
    def test_balancing_get_weights_treed_single_label(self):
        Y = np.array([0] * 80 + [1] * 20)
        balancing = Balancing(strategy='weighting')
        init_params, fit_params = balancing.get_weights(
            Y, 'adaboost', None, None, None)
        self.assertTrue(np.allclose(fit_params['classifier:sample_weight'],
                                    np.array([0.4] * 80 + [1.6] * 20)))
        #init_params, fit_params = balancing.get_weights(
        #    Y, None, 'extra_trees_preproc_for_classification', None, None)
        #self.assertTrue(np.allclose(fit_params['preprocessor:sample_weight'],
        #                            np.array([0.4] * 80 + [1.6] * 20)))

    def test_balancing_get_weights_treed_multilabel(self):
        Y = np.array([[0, 0, 0]] * 100 + [[1, 0, 0]] * 100 + [[0, 1, 0]] * 100 +
                     [[1, 1, 0]] * 100 + [[0, 0, 1]] * 100 + [[1, 0, 1]] * 10)
        balancing = Balancing(strategy='weighting')
        init_params, fit_params = balancing.get_weights(
            Y, 'adaboost', None, None, None)
        self.assertTrue(np.allclose(fit_params['classifier:sample_weight'],
                                    np.array([0.4] * 500 + [4.0] * 10)))
        #init_params, fit_params = balancing.get_weights(
        #    Y, None, 'extra_trees_preproc_for_classification', None, None)
        #self.assertTrue(np.allclose(fit_params['preprocessor:sample_weight'],
        #                            np.array([0.4] * 500 + [4.0] * 10)))

    def test_balancing_get_weights_svm_sgd(self):
        Y = np.array([0] * 80 + [1] * 20)
        balancing = Balancing(strategy='weighting')
        init_params, fit_params = balancing.get_weights(
            Y, 'libsvm_svc', None, None, None)
        self.assertEqual(("classifier:class_weight", "auto"),
                         list(init_params.items())[0])
        init_params, fit_params = balancing.get_weights(
            Y, None, 'liblinear_svc_preprocessor', None, None)
        self.assertEqual(("preprocessor:class_weight", "auto"),
                         list(init_params.items())[0])

    def test_weighting_effect(self):
        for name, clf, acc_no_weighting, acc_weighting in \
                [('adaboost', AdaboostClassifier, 0.692, 0.719),
                 ('decision_tree', DecisionTree, 0.712, 0.668),
                 ('extra_trees', ExtraTreesClassifier, 0.901, 0.919),
                 ('gradient_boosting', GradientBoostingClassifier, 0.879, 0.883),
                 ('random_forest', RandomForest, 0.886, 0.885),
                 ('libsvm_svc', LibSVM_SVC, 0.915, 0.937),
                 ('liblinear_svc', LibLinear_SVC, 0.920, 0.923),
                 ('sgd', SGD, 0.811, 0.902)]:
            for strategy, acc in [('none', acc_no_weighting),
                                  ('weighting', acc_weighting)]:
                # Fit
                X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
                cs = ParamSklearnClassifier.get_hyperparameter_search_space(
                    include={'classifier': [name]})
                default = cs.get_default_configuration()
                default._values['balancing:strategy'] = strategy
                classifier = ParamSklearnClassifier(default, random_state=1)
                predictor = classifier.fit(X_train, Y_train)
                predictions = predictor.predict(X_test)
                self.assertAlmostEqual(acc,
                    sklearn.metrics.accuracy_score(predictions, Y_test),
                    places=3)

                # pre_transform and fit_estimator
                X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
                cs = ParamSklearnClassifier.get_hyperparameter_search_space(
                    include={'classifier': [name]})
                default = cs.get_default_configuration()
                default._values['balancing:strategy'] = strategy
                classifier = ParamSklearnClassifier(default, random_state=1)
                Xt, fit_params = classifier.pre_transform(X_train, Y_train)
                classifier.fit_estimator(Xt, Y_train, fit_params=fit_params)
                predictions = classifier.predict(X_test)
                self.assertAlmostEqual(acc,
                                       sklearn.metrics.accuracy_score(
                                           predictions, Y_test),
                                       places=3)

        for name, pre, acc_no_weighting, acc_weighting in \
                [('extra_trees_preproc_for_classification',
                  ExtraTreesPreprocessor, 0.892, 0.910),
                   ('liblinear_svc_preprocessor', LibLinear_Preprocessor,
                    0.906, 0.909)]:
            for strategy, acc in [('none', acc_no_weighting),
                                  ('weighting', acc_weighting)]:

                X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
                cs = ParamSklearnClassifier.get_hyperparameter_search_space(
                    include={'classifier': ['sgd'], 'preprocessor': [name]})
                default = cs.get_default_configuration()
                default._values['balancing:strategy'] = strategy
                classifier = ParamSklearnClassifier(default, random_state=1)
                predictor = classifier.fit(X_train, Y_train)
                predictions = predictor.predict(X_test)
                self.assertAlmostEqual(acc,
                                       sklearn.metrics.accuracy_score(
                                           predictions, Y_test),
                                       places=3)

                # pre_transform and fit_estimator
                X_train, Y_train, X_test, Y_test = get_dataset(dataset='digits')
                cs = ParamSklearnClassifier.get_hyperparameter_search_space(
                    include={'classifier': ['sgd'], 'preprocessor': [name]})
                default = cs.get_default_configuration()
                default._values['balancing:strategy'] = strategy
                classifier = ParamSklearnClassifier(default, random_state=1)
                Xt, fit_params = classifier.pre_transform(X_train, Y_train)
                classifier.fit_estimator(Xt, Y_train, fit_params=fit_params)
                predictions = classifier.predict(X_test)
                self.assertAlmostEqual(acc,
                                       sklearn.metrics.accuracy_score(
                                           predictions, Y_test),
                                       places=3)