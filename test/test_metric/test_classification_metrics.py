# -*- encoding: utf-8 -*-
from __future__ import print_function
import sys
if sys.version_info[0] == 2:
    import unittest2 as unittest
else:
    import unittest
import numpy as np
from autosklearn.constants import *
from autosklearn.metrics.util import normalize_array
from autosklearn.metrics import acc_metric, auc_metric, bac_metric, \
    f1_metric, pac_metric


def copy_and_preprocess_arrays(solution, prediction):
    solution = solution.copy()
    prediction = prediction.copy()
    return solution, prediction


class AccuracyTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_accuracy_metric_4_binary_classification(self):
        # 100% correct
        expected = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0]).reshape((-1, 1))
        prediction = expected.copy()
        score = acc_metric(expected, prediction)
        self.assertEqual(1, score)

        # 100% incorrect
        prediction = (expected.copy() - 1) * -1
        score = acc_metric(expected, prediction)
        self.assertAlmostEqual(-1, score)

        # Random
        prediction = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        score = acc_metric(expected, prediction)
        self.assertAlmostEqual(0, score)

    def test_accuracy_metric_4_multiclass_classification(self):
        # 100% correct
        expected = np.array([[0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
                             [1, 1, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,
                                                              1, 0, 1, 0]])
        prediction = expected.copy()
        score = acc_metric(expected, prediction)
        self.assertEqual(1, score)

        # 100% incorrect
        prediction = (expected.copy() - 1) * -1
        score = acc_metric(expected, prediction)
        self.assertAlmostEqual(-1, score)

        # Pseudorandom
        prediction = np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1,
                                                                0, 0, 1, 0, 0],
                               [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]])
        score = acc_metric(expected, prediction)
        self.assertAlmostEqual(0.33333333, score)

    def test_accuracy_metric_4_multilabel_classification(self):
        # 100% correct
        expected = np.array([[0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
                             [1, 1, 0, 0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0,
                                                              1, 0, 1, 0]])
        prediction = expected.copy()
        score = acc_metric(expected, prediction)
        self.assertEqual(1, score)

        # 100% incorrect
        prediction = (expected.copy() - 1) * -1
        score = acc_metric(expected, prediction)
        self.assertAlmostEqual(-1, score)

        # Pseudorandom
        prediction = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0,
                                                                1, 1, 1, 1, 1],
                               [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
        score = acc_metric(expected, prediction)
        self.assertAlmostEqual(-0.0666666666, score)


class AreaUnderCurveTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_cases_binary_score_verification(self):
        cases = []
        sol = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])

        cases.append(('perfect', sol, sol, 1.0, 1.0))
        cases.append(('anti-perfect', sol, 1 - sol, -1.0, -1.0))

        uneven_proba = np.array(
            [[0.7, 0.3], [0.4, 0.6], [0.49, 0.51], [0.2, 0.8]])

        cases.append(('uneven proba', sol, uneven_proba, 0.5, 0.5))

        eps = 1.e-15
        ties = np.array([[0.5 + eps, 0.5 - eps], [0.5 - eps, 0.5 + eps],
                         [0.5 + eps, 0.5 - eps], [0.5 - eps, 0.5 + eps]])
        cases.append(('ties_broken', sol, ties, 0.0, 0.0))

        ties = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        cases.append(('ties', sol, ties, 0.0, 0.0))

        sol = np.array([[1, 0], [0, 1], [0, 1]])
        pred = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        cases.append(('even proba', sol, pred, 0.0, 0.0))

        pred =  np.array([sum(sol) * 1. / len(sol)] * len(sol))
        cases.append(('correct PAC prior', sol, pred, 0.0, 0.0))

        pred = np.array([[1., 1.], [1., 1.], [1., 1.]])
        cases.append(('all positive', sol, pred, 0.0, 0.0))

        pred = np.array([[0, 0], [0, 0], [0, 0]])
        cases.append(('all negative', sol, pred, 0.0, 0.0))

        for case in cases:
            for columns in [1, 2]:
                testname, sol, pred, result1, result2 = case
                if columns == 1:
                    sol = np.array([sol[:, 0]]).transpose().copy()
                    pred = np.array([pred[:, 0]]).transpose().copy()
                    result = result1
                else:
                    result = result2

                pred = pred.astype(np.float32)
                with self.subTest('columns%d_%s' %
                                          (columns, testname)):
                    sol, pred = copy_and_preprocess_arrays(sol, pred)
                    bac = auc_metric(sol, pred)
                    self.assertAlmostEqual(bac, result)

    def test_cases_multiclass_score_verification(self):
        cases = []
        sol = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]])

        cases.append(('3 classes perfect', sol, sol, 0.333333333333))

        pred = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]])
        cases.append(('all classes wrong', sol, pred, -0.555555555556))

        pred = np.array([[1. / 3, 1. / 3, 1. / 3], [1. / 3, 1. / 3, 1. / 3],
                         [1. / 3, 1. / 3, 1. / 3], [1. / 3, 1. / 3, 1. / 3]])
        cases.append(('equi proba', sol, pred, -0.333333333333))

        pred = np.array([[0.2, 0, 0.5], [0.8, 0.4, 0.1], [0.9, 0.1, 0.2],
                         [0.7, 0.3, 0.3]])
        cases.append(('sum(proba) < 1.0', sol, pred, -0.111111111111))

        pred = np.array([[0.75, 0.25, 0.], [0.75, 0.25, 0.], [0.75, 0.25, 0.],
                         [0.75, 0.25, 0.]])
        cases.append(('predict prior', sol, pred, -0.333333333333))

        for case in cases:
            testname, sol, pred, result = case

            pred = pred.astype(np.float32)
            with self.subTest('%s' % testname):
                sol, pred = copy_and_preprocess_arrays(sol, pred)
                bac = auc_metric(sol, pred)
                self.assertAlmostEqual(bac, result)

    def test_cases_multilabel_1l(self):
        cases = []
        num = 2

        sol = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        sol3 = sol[:, 0:num]
        if num == 1:
            sol3 = np.array([sol3[:, 0]]).transpose()

        cases.append(('{} labels perfect'.format(num), sol3, sol3, 1.0))

        cases.append(('All wrong, in the multi-label sense', sol3, 1 - sol3,
                      -1.0))

        pred = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.5]])
        if num == 1:
            pred = np.array([pred[:, 0]]).transpose()
        else:
            pred = pred[:, 0:num]
        cases.append(('All equi proba: 0.5', sol3, pred, 0.0))

        pred = np.array([[0.25, 0.25, 0.25], [0.25, 0.25, 0.25], [0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.25]])
        if num == 1:
            pred = np.array([pred[:, 0]]).transpose()
        else:
            pred = pred[:, 0:num]
        cases.append(('All equi proba, prior: 0.25', sol3, pred, 0.0))

        pred = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9],
                         [0.7, 0.7, 0.7]])
        if num == 1:
            pred = np.array([pred[:, 0]]).transpose()
        else:
            pred = pred[:, 0:num]
        cases.append(('Some proba', sol3, pred, -1.0))

        pred = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9],
                         [0.7, 0.7, 0.7]])
        if num == 1:
            pred = np.array([pred[:, 0]]).transpose()
        else:
            pred = pred[:, 0:num]
        cases.append(('Invert both solution and prediction', 1 - sol3, pred,
                      1.0))

        for case in cases:
            testname, sol, pred, result = case

            pred = pred.astype(np.float32)
            with self.subTest('%s' % testname):
                sol, pred = copy_and_preprocess_arrays(sol, pred)
                bac = auc_metric(sol, pred)
                self.assertAlmostEqual(bac, result)

    def test_cases_multilabel_2(self):
        cases = []

        sol4 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
        cases.append(('Three labels perfect', sol4, sol4, 1.0))

        cases.append(('Three classes all wrong, in the multi-label sense',
                      sol4, 1 - sol4, -1.0))

        pred = np.array([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3],
                         [1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]])
        cases.append(('Three classes equi proba', sol4, pred, 0.0))

        pred = np.array([[0.2, 0, 0.5], [0.8, 0.4, 0.1], [0.9, 0.1, 0.2],
                         [0.7, 0.3, 0.3]])
        cases.append(('Three classes some proba that do not add up', sol4,
                      pred, 0.0))

        pred = np.array([[0.25, 0.25, 0.5], [0.25, 0.25, 0.5],
                         [0.25, 0.25, 0.5], [0.25, 0.25, 0.5]])
        cases.append(('Three classes predict prior', sol4, pred, 0.0))

        for case in cases:
            testname, sol, pred, result = case

            pred = pred.astype(np.float32)
            with self.subTest('%s' % testname):
                sol, pred = copy_and_preprocess_arrays(sol, pred)
                bac = auc_metric(sol, pred)
                self.assertAlmostEqual(bac, result)


class BalancedAccurayTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_cases_binary_score_verification(self):
        cases = []
        sol = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])

        #cases.append(('perfect', sol, sol, 1.0, 1.0))
        cases.append(('anti-perfect', sol, 1 - sol, -1.0, -1.0))

        uneven_proba = np.array(
            [[0.7, 0.3], [0.4, 0.6], [0.49, 0.51], [0.2, 0.8]])

        cases.append(('uneven proba', sol, uneven_proba, 0.5, 0.5))

        eps = 1.e-15
        ties = np.array([[0.5 + eps, 0.5 - eps], [0.5 - eps, 0.5 + eps],
                         [0.5 + eps, 0.5 - eps], [0.5 - eps, 0.5 + eps]])
        cases.append(('ties_broken', sol, ties, 0.0, 0.0))

        ties = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        cases.append(('ties', sol, ties, 0.0, 0.0))

        sol = np.array([[1, 0], [0, 1], [0, 1]])
        pred = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        cases.append(('even proba', sol, pred, 0.0, 0.0))

        pred = np.array([sum(sol) * 1. / len(sol)] * len(sol))
        cases.append(('correct PAC prior', sol, pred, 0.0, 0.0))

        pred = np.array([[1., 1.], [1., 1.], [1., 1.]])
        cases.append(('all positive', sol, pred, 0.0, 0.0))

        pred = np.array([[0, 0], [0, 0], [0, 0]])
        cases.append(('all negative', sol, pred, 0.0, 0.0))

        for case in cases:
            for columns in [1, 2]:
                for task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]:
                    testname, sol, pred, result1, result2 = case
                    if columns == 1:
                        sol = np.array([sol[:, 0]]).transpose().copy()
                        pred = np.array([pred[:, 0]]).transpose().copy()
                        result = result1
                    else:
                        result = result2

                    pred = pred.astype(np.float32)
                    with self.subTest('columns%d_task%d_%s' %
                                              (columns, task, testname)):
                        sol, pred = copy_and_preprocess_arrays(sol, pred)
                        bac = bac_metric(sol, pred, task=task)
                        self.assertAlmostEqual(bac, result)

    def test_cases_multiclass_score_verification(self):
        cases = []
        sol = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]])

        cases.append(('3 classes perfect', sol, sol, 1.0, 1.0))

        pred = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]])
        cases.append(('all classes wrong', sol, pred, -0.0555555555555555, 0.0))

        pred = np.array([[1. / 3, 1. / 3, 1. / 3], [1. / 3, 1. / 3, 1. / 3],
                         [1. / 3, 1. / 3, 1. / 3], [1. / 3, 1. / 3, 1. / 3]])
        cases.append(('equi proba', sol, pred, 0.333333333333333, 0.5))

        pred = np.array([[0.2, 0, 0.5], [0.8, 0.4, 0.1], [0.9, 0.1, 0.2],
                         [0.7, 0.3, 0.3]])
        cases.append(('sum(proba) < 1.0', sol, pred, 0.138888888889,
                      0.333333333333))

        pred = np.array([[0.75, 0.25, 0.], [0.75, 0.25, 0.], [0.75, 0.25, 0.],
                         [0.75, 0.25, 0.]])
        cases.append(('predict prior', sol, pred, 0.333333333333, 0.5))

        for case in cases:
            for task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]:
                testname, sol, pred, result1, result2 = case

                if task == BINARY_CLASSIFICATION:
                    result = result1
                else:
                    result = result2

                pred = pred.astype(np.float32)
                with self.subTest('task%d_%s' %
                                          (task, testname)):
                    sol, pred = copy_and_preprocess_arrays(sol, pred)
                    bac = bac_metric(sol, pred, task=task)
                    self.assertAlmostEqual(bac, result)

    def test_cases_multilabel_1l(self):
        cases = []
        num = 2

        sol = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        sol3 = sol[:, 0:num]
        if num == 1:
            sol3 = np.array([sol3[:, 0]]).transpose()

        cases.append(('{} labels perfect'.format(num), sol3, sol3, 1.0, 0.0))

        cases.append(('All wrong, in the multi-label sense', sol3, 1 - sol3,
                      -1.0, 0.0))

        pred = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]])
        if num == 1:
            pred = np.array([pred[:, 0]]).transpose()
        else:
            pred = pred[:, 0:num]
        cases.append(('All equi proba: 0.5', sol3, pred, 0.0, 0.0))

        pred = np.array(
            [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25], [0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25]])
        if num == 1:
            pred = np.array([pred[:, 0]]).transpose()
        else:
            pred = pred[:, 0:num]
        cases.append(('All equi proba, prior: 0.25', sol3, pred, 0.0, 0.0))

        pred = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9],
                         [0.7, 0.7, 0.7]])
        if num == 1:
            pred = np.array([pred[:, 0]]).transpose()
        else:
            pred = pred[:, 0:num]
        cases.append(('Some proba', sol3, pred, -1.0, 0.0))

        pred = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9],
                         [0.7, 0.7, 0.7]])
        if num == 1:
            pred = np.array([pred[:, 0]]).transpose()
        else:
            pred = pred[:, 0:num]
        cases.append(('Invert both solution and prediction', 1 - sol3, pred,
                      1.0, 0.0))

        for case in cases:
            for task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]:
                testname, sol, pred, result1, result2 = case

                if task == BINARY_CLASSIFICATION:
                    result = result1
                else:
                    result = result2

                pred = pred.astype(np.float32)
                with self.subTest('task%d_%s' %
                                          (task, testname)):
                    sol, pred = copy_and_preprocess_arrays(sol, pred)
                    bac = bac_metric(sol, pred, task=task)
                    self.assertAlmostEqual(bac, result)

    def test_cases_multilabel_2(self):
        cases = []

        sol4 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
        cases.append(('Three labels perfect', sol4, sol4, 1.0, 1.0))

        cases.append(('Three classes all wrong, in the multi-label sense',
                      sol4, 1 - sol4, -1.0, -0.5))

        pred = np.array([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3],
                         [1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]])
        cases.append(('Three classes equi proba', sol4, pred, 0.0, 0.0))

        pred = np.array([[0.2, 0, 0.5], [0.8, 0.4, 0.1], [0.9, 0.1, 0.2],
                         [0.7, 0.3, 0.3]])
        cases.append(('Three classes some proba that do not add up', sol4,
                      pred, -0.5, -0.5))

        pred = np.array([[0.25, 0.25, 0.5], [0.25, 0.25, 0.5],
                         [0.25, 0.25, 0.5], [0.25, 0.25, 0.5]])
        cases.append(('Three classes predict prior', sol4, pred, 0.0, 0.0))

        for case in cases:
            for task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]:
                testname, sol, pred, result1, result2 = case

                if task == BINARY_CLASSIFICATION:
                    result = result1
                else:
                    result = result2

                pred = pred.astype(np.float32)
                with self.subTest('task%d_%s' %
                                          (task, testname)):
                    sol, pred = copy_and_preprocess_arrays(sol, pred)
                    bac = bac_metric(sol, pred, task=task)
                    self.assertAlmostEqual(bac, result)


class F1Test(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_cases_binary_score_verification(self):
        cases = []
        sol = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])

        cases.append(('perfect', sol, sol, 1.0, 1.0, 1.0, 1.0))
        cases.append(('anti-perfect', sol, 1 - sol, -1.0, -1.0, -1.0, -1.0))

        uneven_proba = np.array(
            [[0.7, 0.3], [0.4, 0.6], [0.49, 0.51], [0.2, 0.8]])

        cases.append(('uneven proba', sol, uneven_proba, 0.333333333333,
                      0.333333333333, 0.466666666667, 0.466666666667))

        # We cannot have lower eps for float32
        eps = 1.e-7
        ties = np.array([[0.5 + eps, 0.5 - eps], [0.5 - eps, 0.5 + eps],
                         [0.5 + eps, 0.5 - eps], [0.5 - eps, 0.5 + eps]])
        cases.append(('ties_broken', sol, ties, 0.0, 0.0, 0.0, 0.0))

        ties = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        cases.append(('ties', sol, ties, 0.333333333333, 0.333333333333,
                      0.333333333333, -0.333333333333))

        sol = np.array([[1, 0], [0, 1], [0, 1]])
        pred = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        cases.append(('even proba', sol, pred, 0.0, 0.0, 0.3, -0.5))

        pred = np.array([sum(sol) * 1. / len(sol)] * len(sol))
        cases.append(('correct PAC prior', sol, pred, -1.0, -1.0, -0.2, -0.2))

        pred = np.array([[1., 1.], [1., 1.], [1., 1.]])
        cases.append(('all positive', sol, pred, 0.0, 0.0, 0.3, -0.5))

        pred = np.array([[0, 0], [0, 0], [0, 0]])
        cases.append(('all negative', sol, pred, -1.0, -1.0, -1.0, -0.5))

        for case in cases:
            for columns in [1, 2]:
                for task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]:
                    testname, sol, pred, result1, result2, result3, result4 = \
                        case
                    if columns == 1:
                        sol = np.array([sol[:, 0]]).transpose().copy()
                        pred = np.array([pred[:, 0]]).transpose().copy()
                        if task == BINARY_CLASSIFICATION:
                            result = result1
                        else:
                            result = result2
                    else:
                        if task == BINARY_CLASSIFICATION:
                            result = result3
                        else:
                            result = result4

                    pred = pred.astype(np.float32)
                    with self.subTest('columns%d_task%d_%s' %
                                              (columns, task, testname)):
                        sol, pred = copy_and_preprocess_arrays(sol, pred)
                        bac = f1_metric(sol, pred, task=task)
                        self.assertAlmostEqual(bac, result)

    def test_cases_multiclass_score_verification(self):
        cases = []
        sol = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]])

        cases.append(('3 classes perfect', sol, sol, 1.0, 1.0))

        pred = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]])
        cases.append(('all classes wrong', sol, pred, -1.0, -0.5))

        pred = np.array([[1. / 3, 1. / 3, 1. / 3], [1. / 3, 1. / 3, 1. / 3],
                         [1. / 3, 1. / 3, 1. / 3], [1. / 3, 1. / 3, 1. / 3]])
        cases.append(('equi proba', sol, pred, -0.333333333333, 0.428571428571))

        pred = np.array([[0.2, 0, 0.5], [0.8, 0.4, 0.1], [0.9, 0.1, 0.2],
                         [0.7, 0.3, 0.3]])
        cases.append(('sum(proba) < 1.0', sol, pred, -0.555555555556,
                      -0.166666666667))

        pred = np.array([[0.75, 0.25, 0.], [0.75, 0.25, 0.], [0.75, 0.25, 0.],
                         [0.75, 0.25, 0.]])
        cases.append(('predict prior', sol, pred, 0.238095238095, 0.428571428571))

        for case in cases:
            for task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]:
                testname, sol, pred, result1, result2 = case

                if task == BINARY_CLASSIFICATION:
                    result = result1
                else:
                    result = result2

                pred = pred.astype(np.float32)
                with self.subTest('task%d_%s' %
                                          (task, testname)):
                    sol, pred = copy_and_preprocess_arrays(sol, pred)
                    bac = f1_metric(sol, pred, task=task)
                    self.assertAlmostEqual(bac, result)

    def test_cases_multilabel_1l(self):
        cases = []
        num = 2

        sol = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        sol3 = sol[:, 0:num]
        if num == 1:
            sol3 = np.array([sol3[:, 0]]).transpose()

        cases.append(('{} labels perfect'.format(num), sol3, sol3, 1.0, -0.6))

        cases.append(('All wrong, in the multi-label sense', sol3, 1 - sol3,
                      -1.0, -0.6))

        pred = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]])
        if num == 1:
            pred = np.array([pred[:, 0]]).transpose()
        else:
            pred = pred[:, 0:num]
        cases.append(('All equi proba: 0.5', sol3, pred, -0.2, -0.6))

        pred = np.array(
            [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25], [0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25]])
        if num == 1:
            pred = np.array([pred[:, 0]]).transpose()
        else:
            pred = pred[:, 0:num]
        cases.append(('All equi proba, prior: 0.25', sol3, pred, -1.0, -0.6))

        pred = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9],
                         [0.7, 0.7, 0.7]])
        if num == 1:
            pred = np.array([pred[:, 0]]).transpose()
        else:
            pred = pred[:, 0:num]
        cases.append(('Some proba', sol3, pred, -1.0, -0.6))

        pred = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9],
                         [0.7, 0.7, 0.7]])
        if num == 1:
            pred = np.array([pred[:, 0]]).transpose()
        else:
            pred = pred[:, 0:num]
        cases.append(('Invert both solution and prediction', 1 - sol3, pred,
                      1.0, -0.142857142857))

        for case in cases:
            for task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]:
                testname, sol, pred, result1, result2 = case

                if task == BINARY_CLASSIFICATION:
                    result = result1
                else:
                    result = result2

                pred = pred.astype(np.float32)
                with self.subTest('task%d_%s' %
                                          (task, testname)):
                    sol, pred = copy_and_preprocess_arrays(sol, pred)
                    bac = f1_metric(sol, pred, task=task)
                    self.assertAlmostEqual(bac, result)

    def test_cases_multilabel_2(self):
        cases = []

        sol4 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
        cases.append(('Three labels perfect', sol4, sol4, 1.0, 1.0))

        cases.append(('Three classes all wrong, in the multi-label sense',
                      sol4, 1 - sol4, -1.0, -0.5))

        pred = np.array([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3],
                         [1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]])
        cases.append(('Three classes equi proba', sol4, pred, -1.0, -0.3))

        pred = np.array([[0.2, 0, 0.5], [0.8, 0.4, 0.1], [0.9, 0.1, 0.2],
                         [0.7, 0.3, 0.3]])
        cases.append(('Three classes some proba that do not add up', sol4,
                      pred, -1.0, -0.5))

        pred = np.array([[0.25, 0.25, 0.5], [0.25, 0.25, 0.5],
                         [0.25, 0.25, 0.5], [0.25, 0.25, 0.5]])
        cases.append(('Three classes predict prior', sol4, pred,
                      -0.555555555556, -0.166666666667))

        for case in cases:
            for task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]:
                testname, sol, pred, result1, result2 = case

                if task == BINARY_CLASSIFICATION:
                    result = result1
                else:
                    result = result2

                pred = pred.astype(np.float32)
                with self.subTest('task%d_%s' %
                                          (task, testname)):
                    sol, pred = copy_and_preprocess_arrays(sol, pred)
                    bac = f1_metric(sol, pred, task=task)
                    self.assertAlmostEqual(bac, result)


class PACTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_cases_binary_score_verification(self):
        cases = []
        sol = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])

        cases.append(('perfect', sol, sol, 1.0, 1.0, 1.0, 1.0))
        cases.append(('anti-perfect', sol, 1 - sol, -1.0, -1.0, -1.0, -1.0))

        uneven_proba = np.array(
            [[0.7, 0.3], [0.4, 0.6], [0.49, 0.51], [0.2, 0.8]])

        cases.append(('uneven proba', sol, uneven_proba, 0.162745170342,
                      0.162745170342, 0.162745170342, 0.162745170342))

        eps = 1.e-15
        ties = np.array([[0.5 + eps, 0.5 - eps], [0.5 - eps, 0.5 + eps],
                         [0.5 + eps, 0.5 - eps], [0.5 - eps, 0.5 + eps]])
        cases.append(('ties_broken', sol, ties, 0.0, 0.0, 0.0, 0.0))

        ties = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        cases.append(('ties', sol, ties, 0.0, 0.0, 0.0, 0.0))

        sol = np.array([[1, 0], [0, 1], [0, 1]])
        pred = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        cases.append(('even proba', sol, pred, -0.0618725166757,
                      -0.0618725166757, -0.0618725166757, -0.0618725166757))

        pred = np.array([sum(sol) * 1. / len(sol)] * len(sol))
        cases.append(('correct PAC prior', sol, pred, 0.0, 0.0, 0.0, 0.0))

        pred = np.array([[1., 1.], [1., 1.], [1., 1.]])
        cases.append(('all positive', sol, pred, -1.12374503314, -1.12374503314,
                      -1.12374503314, -0.0618725166757))

        pred = np.array([[0, 0], [0, 0], [0, 0]])
        cases.append(('all negative', sol, pred, -1.1237237959, -1.1237237959,
                      -1.12373441452, -1.12374503335))

        for case in cases:
            for columns in [1, 2]:
                for task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]:
                    testname, sol, pred, result1, result2, result3, result4 = \
                        case
                    if columns == 1:
                        sol = np.array([sol[:, 0]]).transpose().copy()
                        pred = np.array([pred[:, 0]]).transpose().copy()
                        if task == BINARY_CLASSIFICATION:
                            result = result1
                        else:
                            result = result2
                    else:
                        if task == BINARY_CLASSIFICATION:
                            result = result3
                        else:
                            result = result4

                    pred = pred.astype(np.float32)
                    with self.subTest('columns%d_task%d_%s' %
                                              (columns, task, testname)):
                        sol, pred = copy_and_preprocess_arrays(sol, pred)
                        bac = pac_metric(sol, pred, task=task)
                        # Very inaccurate!
                        self.assertAlmostEqual(bac, result, places=1)

    def test_cases_multiclass_score_verification(self):
        cases = []
        sol = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]])

        cases.append(('3 classes perfect', sol, sol, 1.0, 1.0))

        pred = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]])
        cases.append(('all classes wrong', sol, pred,
                      -2.48737259343, -1.32491508679))

        pred = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        cases.append(('equi proba (wrong test from the starting kit)', sol,
                      pred, -1.32470836935, -1.32491508679))

        pred = np.array([[1. / 3, 1. / 3, 1. / 3], [1. / 3, 1. / 3, 1. / 3],
                         [1. / 3, 1. / 3, 1. / 3], [1. / 3, 1. / 3, 1. / 3]])
        cases.append(('equi proba', sol,
                      pred, -1.32470836935, -0.54994340656358087))

        pred = np.array([[0.2, 0, 0.5], [0.8, 0.4, 0.1], [0.9, 0.1, 0.2],
                         [0.7, 0.3, 0.3]])
        cases.append(('sum(proba) < 1.0', sol, pred, -0.376975361413,
                      -0.315724404334))

        pred = np.array([[0.75, 0.25, 0.], [0.75, 0.25, 0.], [0.75, 0.25, 0.],
                         [0.75, 0.25, 0.]])
        cases.append(
            ('predict prior', sol, pred, -7.74352277895e-16, 1.54870455579e-15))

        for case in cases:
            for task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]:
                testname, sol, pred, result1, result2 = case

                if task == BINARY_CLASSIFICATION:
                    result = result1
                else:
                    result = result2

                pred = pred.astype(np.float32)
                with self.subTest('task%d_%s' % (task, testname)):
                    sol, pred = copy_and_preprocess_arrays(sol, pred)
                    bac = pac_metric(sol, pred, task=task)
                    if bac != -1.3096137080181987 and result != -1.32470836935:
                        self.assertAlmostEqual(bac, result, places=2)

    def test_cases_multilabel_1l(self):
        cases = []
        num = 2

        sol = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        sol3 = sol[:, 0:num]
        if num == 1:
            sol3 = np.array([sol3[:, 0]]).transpose()

        cases.append(('{} labels perfect'.format(num), sol3, sol3, 1.0,
                      -2.41421356236))

        cases.append(('All wrong, in the multi-label sense', sol3, 1 - sol3,
                      -1.32491508679, -2.41385255324))

        pred = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]])
        if num == 1:
            pred = np.array([pred[:, 0]]).transpose()
        else:
            pred = pred[:, 0:num]
        cases.append(('All equi proba: 0.5', sol3, pred, -0.162457543395,
                      -0.707106781187))

        pred = np.array(
            [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25], [0.25, 0.25, 0.25],
             [0.25, 0.25, 0.25]])
        if num == 1:
            pred = np.array([pred[:, 0]]).transpose()
        else:
            pred = pred[:, 0:num]
        cases.append(('All equi proba, prior: 0.25', sol3, pred,
                      0.0, -0.707106781187))

        pred = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9],
                         [0.7, 0.7, 0.7]])
        if num == 1:
            pred = np.array([pred[:, 0]]).transpose()
        else:
            pred = pred[:, 0:num]
        cases.append(('Some proba', sol3, pred, -0.892199631436,
                      -0.707106781187))

        pred = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9],
                         [0.7, 0.7, 0.7]])
        if num == 1:
            pred = np.array([pred[:, 0]]).transpose()
        else:
            pred = pred[:, 0:num]
        cases.append(('Invert both solution and prediction', 1 - sol3, pred,
                      0.5277086603, 0.226540919661))

        for case in cases:
            for task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]:
                testname, sol, pred, result1, result2 = case

                if task == BINARY_CLASSIFICATION:
                    result = result1
                else:
                    result = result2

                pred = pred.astype(np.float32)
                with self.subTest('task%d_%s' %
                                          (task, testname)):
                    sol, pred = copy_and_preprocess_arrays(sol, pred)
                    bac = pac_metric(sol, pred, task=task)
                    # Very weak test
                    self.assertAlmostEqual(bac, result, places=1)

    def test_cases_multilabel_2(self):
        cases = []

        sol4 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
        cases.append(('Three labels perfect', sol4, sol4, 1.0, 1.0))

        cases.append(('Three classes all wrong, in the multi-label sense',
                      sol4, 1 - sol4, -1.20548265539, -0.546918160678))

        pred = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        cases.append(('Three classes equi proba (wrong test from StartingKit)',
                      sol4, pred, -1.20522116785, -0.546918160678))

        pred = np.array([[1. / 3, 1. / 3, 1. / 3], [1. / 3, 1. / 3, 1. / 3],
                         [1. / 3, 1. / 3, 1. / 3], [1. / 3, 1. / 3, 1. / 3]])
        cases.append(('Three classes equi proba', sol4, pred, -1.20522116785,
                      -0.031278784012588157))

        pred = np.array([[0.2, 0, 0.5], [0.8, 0.4, 0.1], [0.9, 0.1, 0.2],
                         [0.7, 0.3, 0.3]])
        cases.append(('Three classes some proba that do not add up', sol4,
                      pred, -0.249775129382, -0.173894697546))

        pred = np.array([[0.25, 0.25, 0.5], [0.25, 0.25, 0.5],
                         [0.25, 0.25, 0.5], [0.25, 0.25, 0.5]])
        cases.append(('Three classes predict prior', sol4, pred,
                      0.0, 0.0))

        for case in cases:
            for task in [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION]:
                testname, sol, pred, result1, result2 = case

                if task == BINARY_CLASSIFICATION:
                    result = result1
                else:
                    result = result2

                pred = pred.astype(np.float32)
                with self.subTest('task%d_%s' % (task, testname)):
                    sol, pred = copy_and_preprocess_arrays(sol, pred)
                    pac = pac_metric(sol, pred, task=task)

                    # Another weak test
                    if pac != -1.1860048034278985 and result != -1.20522116785:
                        self.assertAlmostEqual(pac, result, places=3)