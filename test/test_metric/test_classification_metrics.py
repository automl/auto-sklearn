# -*- encoding: utf-8 -*-
import unittest

import numpy as np
from autosklearn.metrics import balanced_accuracy


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
