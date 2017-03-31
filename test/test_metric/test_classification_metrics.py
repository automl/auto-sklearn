# -*- encoding: utf-8 -*-
import sys
import unittest

import numpy as np
from autosklearn.constants import *
from autosklearn.metrics import balanced_accuracy


def copy_and_preprocess_arrays(solution, prediction):
    solution = solution.copy()
    prediction = prediction.copy()
    return solution, prediction


class BalancedAccurayTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_cases_binary_score_verification(self):
        cases = []
        sol = np.array([0, 0, 1, 1])
        pred = np.array([0, 0, 1, 1])

        cases.append(('perfect', sol, pred, 1.0))
        cases.append(('anti-perfect', sol, 1 - pred, -1.0,))

        uneven_proba = np.array(
            [[0.7, 0.3], [0.4, 0.6], [0.49, 0.51], [0.2, 0.8]])
        uneven_proba = np.argmax(uneven_proba, axis=1)

        cases.append(('uneven proba', sol, uneven_proba, 0.5))

        eps = 1.e-15
        ties = np.array([[0.5 + eps, 0.5 - eps], [0.5 - eps, 0.5 + eps],
                         [0.5 + eps, 0.5 - eps], [0.5 - eps, 0.5 + eps]])
        ties = np.argmax(ties, axis=1)
        cases.append(('ties_broken', sol, ties, 0.0))

        ties = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        ties = np.argmax(ties, axis=1)
        cases.append(('ties', sol, ties, 0.0))

        sol = np.array([0, 1, 1])
        pred = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        pred = np.argmax(pred, axis=1)
        cases.append(('even proba', sol, pred, 0.0))

        _pred = np.array([[1, 0], [0, 1], [0, 1]])
        pred = np.array([sum(_pred) * 1. / len(_pred)] * len(_pred))
        pred = np.argmax(pred, axis=1)
        cases.append(('correct PAC prior', sol, pred, 0.0))

        pred = np.array([[1., 1.], [1., 1.], [1., 1.]])
        pred = np.argmax(pred, axis=1)
        cases.append(('all positive', sol, pred, 0.0))

        pred = np.array([[0, 0], [0, 0], [0, 0]])
        pred = np.argmax(pred, axis=1)
        cases.append(('all negative', sol, pred, 0.0))

        for case in cases:
            testname, sol, pred, result = case

            pred = pred.astype(np.float32)
            with self.subTest('%s' % testname):
                sol, pred = copy_and_preprocess_arrays(sol, pred)
                bac = balanced_accuracy(sol, pred)
                self.assertAlmostEqual(bac, result)

    def test_cases_multiclass_score_verification(self):
        cases = []
        sol = np.array([0, 1, 0, 0])
        pred = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]])

        cases.append(('3 classes perfect', sol, pred, 1.0))

        cases.append(('all classes wrong', sol, 1 - pred, 0.0))

        pred = np.array([[1. / 3, 1. / 3, 1. / 3], [1. / 3, 1. / 3, 1. / 3],
                         [1. / 3, 1. / 3, 1. / 3], [1. / 3, 1. / 3, 1. / 3]])
        cases.append(('equi proba', sol, pred, 0.5))

        pred = np.array([[0.2, 0, 0.5], [0.8, 0.4, 0.1], [0.9, 0.1, 0.2],
                         [0.7, 0.3, 0.3]])
        cases.append(('sum(proba) < 1.0', sol, pred, 0.333333333333))

        pred = np.array([[0.75, 0.25, 0.], [0.75, 0.25, 0.], [0.75, 0.25, 0.],
                         [0.75, 0.25, 0.]])
        cases.append(('predict prior', sol, pred, 0.5))

        for case in cases:
            testname, sol, pred, result = case

            pred = pred.astype(np.float32)
            with self.subTest('%s' % testname):
                sol, pred = copy_and_preprocess_arrays(sol, pred)
                bac = balanced_accuracy(sol, pred, task=MULTICLASS_CLASSIFICATION)
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

        pred = np.array(
            [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25], [0.25, 0.25, 0.25],
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
                bac = balanced_accuracy(sol, pred, task=MULTILABEL_CLASSIFICATION)
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
                      pred, -0.5))

        pred = np.array([[0.25, 0.25, 0.5], [0.25, 0.25, 0.5],
                         [0.25, 0.25, 0.5], [0.25, 0.25, 0.5]])
        cases.append(('Three classes predict prior', sol4, pred, 0.0))

        for case in cases:
            testname, sol, pred, result = case

            pred = pred.astype(np.float32)
            with self.subTest('_%s' % testname):
                sol, pred = copy_and_preprocess_arrays(sol, pred)
                bac = balanced_accuracy(sol, pred, task=MULTILABEL_CLASSIFICATION)
                self.assertAlmostEqual(bac, result)
