# -*- encoding: utf-8 -*-
from __future__ import print_function
import copy
import multiprocessing
import os
import sys
import shutil
import unittest

import numpy as np

from autosklearn.constants import *
from autosklearn.evaluation import CVEvaluator, eval_partial_cv, eval_cv
from autosklearn.util import backend
from autosklearn.util.pipeline import get_configuration_space

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import get_dataset_getters, BaseEvaluatorTest, \
    get_multiclass_classification_datamanager

N_TEST_RUNS = 3


class CVEvaluator_Test(BaseEvaluatorTest):
    _multiprocess_can_split_ = True

    def test_datasets(self):
        for getter in get_dataset_getters():
            testname = '%s_%s' % (os.path.basename(__file__).
                                  replace('.pyc', '').replace('.py', ''),
                                  getter.__name__)
            with self.subTest(testname):
                D = getter()
                output_directory = os.path.join(os.path.dirname(__file__),
                                                '.%s' % testname)
                self.output_directories.append(output_directory)
                err = np.zeros([N_TEST_RUNS])
                for i in range(N_TEST_RUNS):
                    D_ = copy.deepcopy(D)
                    evaluator = CVEvaluator(D_, output_directory, None)

                    err[i] = evaluator.fit_predict_and_loss()[0]

                    self.assertTrue(np.isfinite(err[i]))
                    self.assertEqual(err[i], 1.0)
                    for model_idx in range(10):
                        indices = evaluator.indices[model_idx]
                        self.assertIsNotNone(indices)

                    D_ = copy.deepcopy(D)
                    evaluator = CVEvaluator(D_, output_directory, None)
                    for j in range(5):
                        evaluator.partial_fit_predict_and_loss(j)
                        indices = evaluator.indices[j]
                        self.assertIsNotNone(indices)
                    for j in range(5, 10):
                        indices = evaluator.indices[j]
                        self.assertIsNone(indices)

            for i in range(5):
                try:
                    shutil.rmtree(output_directory)
                except Exception:
                    pass


class FunctionsTest(unittest.TestCase):
    def setUp(self):
        self.queue = multiprocessing.Queue()
        self.configuration = get_configuration_space(
            {'task': MULTICLASS_CLASSIFICATION,
             'is_sparse': False}).get_default_configuration()
        self.data = get_multiclass_classification_datamanager()
        self.tmp_dir = os.path.join(os.path.dirname(__file__),
                                    '.test_cv_functions')

    def tearDown(self):
        try:
            shutil.rmtree(self.tmp_dir)
        except Exception:
            pass

    def test_eval_cv(self):
        backend_api = backend.create(self.tmp_dir, self.tmp_dir)
        eval_cv(queue=self.queue, config=self.configuration, data=self.data,
                backend=backend_api, seed=1, num_run=1, folds=5, subsample=None,
                with_predictions=True, all_scoring_functions=False,
                output_y_test=True, include=None, exclude=None)
        info = self.queue.get()
        self.assertAlmostEqual(info[1], 0.095262096774193505)
        self.assertEqual(info[2], 1)
        self.assertNotIn('bac_metric', info[3])

    def test_eval_cv_all_loss_functions(self):
        backend_api = backend.create(self.tmp_dir, self.tmp_dir)
        eval_cv(queue=self.queue, config=self.configuration, data=self.data,
                backend=backend_api, seed=1, num_run=1, folds=5, subsample=None,
                with_predictions=True, all_scoring_functions=True,
                output_y_test=True, include=None, exclude=None)
        info = self.queue.get()
        self.assertIn(
            'f1_metric: 0.0952620967742;pac_metric: 0.355606202593;'
            'acc_metric: 0.09;auc_metric: 0.030994474695;'
            'bac_metric: 0.0952620967742;duration: ', info[3])
        self.assertAlmostEqual(info[1], 0.095262096774193505)
        self.assertEqual(info[2], 1)

    def test_eval_cv_on_subset(self):
        backend_api = backend.create(self.tmp_dir, self.tmp_dir)
        eval_cv(queue=self.queue, config=self.configuration, data=self.data,
                backend=backend_api, seed=1, num_run=1, folds=5, subsample=45,
                with_predictions=True, all_scoring_functions=False,
                output_y_test=True, include=None, exclude=None)
        info = self.queue.get()
        self.assertAlmostEqual(info[1], 0.063004032258064502)
        self.assertEqual(info[2], 1)

    def test_eval_partial_cv(self):
        results = [0.071428571428571508,
                   0.15476190476190488,
                   0.08333333333333337,
                   0.16666666666666674,
                   0.0]
        for fold in range(5):
            backend_api = backend.create(self.tmp_dir, self.tmp_dir)
            eval_partial_cv(queue=self.queue, config=self.configuration,
                            data=self.data, backend=backend_api, seed=1,
                            num_run=1, instance=fold, folds=5,
                            subsample=None, with_predictions=True,
                            all_scoring_functions=False, output_y_test=True,
                            include=None, exclude=None)
            info = self.queue.get()
            results.append(info[1])
            self.assertAlmostEqual(info[1], results[fold])
            self.assertEqual(info[2], 1)

    def test_eval_partial_cv_on_subset_no_timeout(self):
        backend_api = backend.create(self.tmp_dir, self.tmp_dir)

        results = [0.071428571428571508,
                   0.15476190476190488,
                   0.08333333333333337,
                   0.16666666666666674,
                   0.0]
        for fold in range(5):
            eval_partial_cv(queue=self.queue, config=self.configuration,
                            data=self.data, backend=backend_api,
                            seed=1, num_run=1, instance=fold, folds=5,
                            subsample=80, with_predictions=True,
                            all_scoring_functions=False, output_y_test=True,
                            include=None, exclude=None)

            info = self.queue.get()
            self.assertAlmostEqual(info[1], results[fold])
            self.assertEqual(info[2], 1)

        results = [0.071428571428571508,
                   0.15476190476190488,
                   0.16666666666666674,
                   0.0,
                   0.0]
        for fold in range(5):
            eval_partial_cv(queue=self.queue, config=self.configuration,
                            data=self.data, backend=backend_api,
                            seed=1, num_run=1, instance=fold, folds=5,
                            subsample=43, with_predictions=True,
                            all_scoring_functions=False, output_y_test=True,
                            include=None, exclude=None)

            info = self.queue.get()
            self.assertAlmostEqual(info[1], results[fold])
            self.assertEqual(info[2], 1)