# -*- encoding: utf-8 -*-
import unittest

import numpy as np
from autosklearn.metrics import balanced_accuracy, pac_score


def copy_and_preprocess_arrays(solution, prediction):
    solution = solution.copy()
    prediction = prediction.copy()
    return solution, prediction


class BalancedAccurayTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def _test_cases(self, cases):
        for case in cases:
            testname, sol, pred, result = case

            pred = pred.astype(np.float32)
            sol, pred = copy_and_preprocess_arrays(sol, pred)
            bac = balanced_accuracy(sol, pred)
            self.assertAlmostEqual(bac, result, msg=testname)

    def test_binary_balanced(self):
        cases = []
        sol = np.array([0, 0, 1, 1])
        pred = np.array([0, 0, 1, 1])

        cases.append(('perfect', sol, pred, 1.0))
        cases.append(('completely wrong', sol, 1 - pred, 0.0))

        pred = np.array([0, 1, 1, 1])
        cases.append(('partially correct 1', sol, pred, 0.75))

        pred = np.array([0, 1, 0, 1])
        cases.append(('partially correct 2', sol, pred, 0.5))

        pred = np.array([0, 1, 0, 0])
        cases.append(('partially correct 3', sol, pred, 0.25))

        self._test_cases(cases)

    def test_binary_imbalanced(self):
        cases = []
        sol = np.array([0, 1, 1])
        pred = np.array([0, 1, 1])

        cases.append(('perfect', sol, pred, 1.0))
        cases.append(('completely wrong', sol, 1 - pred, 0.0))

        pred = np.array([0, 0, 0])
        cases.append(('one class correct', sol, pred, 0.5))

        pred = np.array([0, 1, 0])
        cases.append(('one class correct, one partially correct',
                      sol, pred, 0.75))

        pred = np.array([1, 0, 1])
        cases.append(('one class partially correct', sol, pred, 0.25))

        self._test_cases(cases)

    def test_multiclass_balanced(self):
        cases = []
        sol = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([0, 0, 1, 1, 2, 2])

        cases.append(('perfect', sol, pred, 1.0))

        pred = np.array([1, 1, 2, 2, 0, 0])
        cases.append(('completely wrong', sol, pred, 0.0))

        pred = np.array([0, 0, 0, 0, 0, 0])
        cases.append(('one class correct', sol, pred, 0.33333333))

        pred = np.array([0, 0, 1, 1, 0, 0])
        cases.append(('two classes correct', sol, pred, 0.66666666))

        pred = np.array([0, 0, 1, 0, 2, 0])
        cases.append(('one class correct, two partially correct', sol, pred, 0.66666666))

        pred = np.array([0, 1, 1, 2, 2, 0])
        cases.append(('all partially correct', sol, pred, 0.5))
        self._test_cases(cases)

    def test_multiclass_imbalanced(self):
        cases = []
        sol = np.array([0, 1, 2, 0])
        pred = np.array([0, 1, 2, 0])

        cases.append(('all classes perfect', sol, pred, 1.0))

        pred = np.array([1, 2, 0, 1])
        cases.append(('all classes wrong', sol, pred, 0.0))

        pred = np.array([0, 0, 0, 0])
        cases.append(('one class correct', sol, pred, 0.33333333))

        pred = np.array([2, 0, 0, 0])
        cases.append(('one class half-correct', sol, pred, 0.16666666))

        self._test_cases(cases)

    def test_multilabel_balanced(self):
        cases = []
        sol = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
        pred = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])

        cases.append(('perfect', sol, pred, 1.0))
        cases.append(('completely wrong', sol, 1 - pred, 0.0))

        pred = np.array([[0, 0], [0, 0], [0, 0], [1, 1]])
        cases.append(('one sample per label wrong', sol, pred, 0.75))

        pred = np.array([[0, 0], [0, 0], [0, 1], [1, 1]])
        cases.append(('one sample in one label wrong', sol, pred, 0.875))

        pred = np.array([[0, 0], [0, 0], [0, 1], [0, 1]])
        cases.append(('two samples in one label wrong', sol, pred, 0.75))
        self._test_cases(cases)

    def test_cases_multilabel(self):
        cases = []
        num = 2

        sol = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        sol3 = sol[:, 0:num]
        if num == 1:
            sol3 = np.array([sol3[:, 0]]).transpose()

        cases.append(('{} labels perfect'.format(num), sol3, sol3, 1.0))

        cases.append(('All wrong, in the multi-label sense', sol3, 1 - sol3,
                      0.0))

        sol4 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
        cases.append(('Three labels perfect', sol4, sol4, 1.0))

        cases.append(('Three classes all wrong, in the multi-label sense',
                      sol4, 1 - sol4, 0.0))

        self._test_cases(cases)


class PACTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def _test_cases(self, cases):
        for case in cases:
            testname, sol, pred, result = case

            pred = pred.astype(np.float32)
            sol, pred = copy_and_preprocess_arrays(sol, pred)
            pac = pac_score(sol, pred)
            self.assertAlmostEqual(pac, result, msg=testname, places=1)

    def test_cases_binary_score_verification(self):
        cases = []
        sol = np.array([0, 0, 1, 1])
        pred = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])

        cases.append(('perfect', sol, pred, 1.0))
        cases.append(('anti-perfect', sol, 1 - pred, -1.0,))

        uneven_proba = np.array(
            [[0.7, 0.3], [0.4, 0.6], [0.49, 0.51], [0.2, 0.8]])

        cases.append(('uneven proba', sol, uneven_proba, 0.162745170342))

        eps = 1.e-15
        ties = np.array([[0.5 + eps, 0.5 - eps], [0.5 - eps, 0.5 + eps],
                         [0.5 + eps, 0.5 - eps], [0.5 - eps, 0.5 + eps]])
        cases.append(('ties_broken', sol, ties, 0.0))

        ties = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        cases.append(('ties', sol, ties, 0.0))

        sol = np.array([0, 1, 1])
        pred = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        cases.append(('even proba', sol, pred, -0.0618725166757))

        _pred = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        pred = np.array([sum(_pred) * 1. / len(_pred)] * len(_pred))
        cases.append(('correct PAC prior', sol, pred, 0.0))

        pred = np.array([[1., 1.], [1., 1.], [1., 1.]])
        cases.append(('all positive', sol, pred, -1.12374503314))

        pred = np.array([[0, 0], [0, 0], [0, 0]])
        cases.append(('all negative', sol, pred, -1.1237237959))

        self._test_cases(cases)

    def test_cases_multiclass_score_verification(self):
        cases = []
        sol = np.array([0, 1, 2, 0])
        pred = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2], [1, 0, 0]])

        cases.append(('3 classes perfect', sol, pred, 1.0))

        pred = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]])
        cases.append(('all classes wrong', sol, pred, -0.5469181142705154))

        pred = np.array([[0., 0., 0.]] * 4)
        cases.append(('equi proba (wrong test from the starting kit)',
                      sol, pred, -0.5469181142705154))

        pred = np.array([[1. / 3, 1. / 3, 1. / 3], [1. / 3, 1. / 3, 1. / 3],
                         [1. / 3, 1. / 3, 1. / 3], [1. / 3, 1. / 3, 1. / 3]])
        cases.append(('equi proba', sol, pred, -0.031278784012588157))

        pred = np.array([[0.2, 0, 0.5], [0.8, 0.4, 0.1], [0.9, 0.1, 0.2],
                         [0.7, 0.3, 0.3]])
        cases.append(('sum(proba) < 1.0', sol, pred, -0.085886926180064257))

        pred = np.array([[0.5, 0.25, 0.25], [0.5, 0.25, 0.25],
                         [0.5, 0.25, 0.25], [0.5, 0.25, 0.25]])
        cases.append(('predict prior', sol, pred, 0))

        self._test_cases(cases)

    def test_cases_multilabel_1l(self):
        cases = []
        num = 2

        sol = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        sol3 = sol[:, 0:num]
        if num == 1:
            sol3 = np.array([sol3[:, 0]]).transpose()

        cases.append(('{} labels perfect'.format(num), sol3, sol3, 1.0))

        cases.append(('All wrong, in the multi-label sense', sol3, 1 - sol3,
                      -1.32491508679))

        pred = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]])
        if num == 1:
            pred = np.array([pred[:, 0]]).transpose()
        else:
            pred = pred[:, 0:num]
        cases.append(('All equi proba: 0.5', sol3, pred, -0.162457543395))

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
        cases.append(('Some proba', sol3, pred, -0.892199631436))

        pred = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8], [0.9, 0.9, 0.9],
                         [0.7, 0.7, 0.7]])
        if num == 1:
            pred = np.array([pred[:, 0]]).transpose()
        else:
            pred = pred[:, 0:num]
        cases.append(('Invert both solution and prediction', 1 - sol3, pred,
                      0.5277086603))

        self._test_cases(cases)

    def test_cases_multilabel_2(self):
        cases = []

        sol4 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
        cases.append(('Three labels perfect', sol4, sol4, 1.0))

        cases.append(('Three classes all wrong, in the multi-label sense',
                      sol4, 1 - sol4, -1.20548265539))

        # Not at random because different classes have different priors
        pred = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        cases.append(('Three classes equi proba (wrong test from StartingKit)',
                      sol4, pred, -1.20522116785))

        pred = np.array([[1. / 3, 1. / 3, 1. / 3], [1. / 3, 1. / 3, 1. / 3],
                         [1. / 3, 1. / 3, 1. / 3], [1. / 3, 1. / 3, 1. / 3]])
        cases.append(('Three classes equi proba', sol4, pred,
                      -0.034665665346400684))

        pred = np.array([[0.2, 0, 0.5], [0.8, 0.4, 0.1], [0.9, 0.1, 0.2],
                         [0.7, 0.3, 0.3]])
        cases.append(('Three classes some proba that do not add up', sol4,
                      pred, -0.249775129382))

        pred = np.array([[0.25, 0.25, 0.5], [0.25, 0.25, 0.5],
                         [0.25, 0.25, 0.5], [0.25, 0.25, 0.5]])
        cases.append(('Three classes predict prior', sol4, pred, 0.0))

        self._test_cases(cases)