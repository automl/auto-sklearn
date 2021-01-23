# -*- encoding: utf-8 -*-
import copy
import json
import logging.handlers
import multiprocessing
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock

import numpy as np
from smac.tae import StatusType

from autosklearn.constants import MULTILABEL_CLASSIFICATION, BINARY_CLASSIFICATION, \
    MULTICLASS_CLASSIFICATION, REGRESSION
from autosklearn.evaluation.test_evaluator import TestEvaluator, eval_t
from autosklearn.evaluation.util import read_queue
from autosklearn.util.pipeline import get_configuration_space
from autosklearn.util.backend import Backend
from autosklearn.metrics import accuracy, r2, f1_macro

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import (  # noqa (E402: module level import not at top of file)
    get_evaluation_backend,
    get_dataset_getters,
    BaseEvaluatorTest,
    get_multiclass_classification_datamanager,
    SCORER_LIST
)  # noqa (E402: module level import not at top of file)


N_TEST_RUNS = 3


class Dummy(object):
    pass


class TestEvaluator_Test(BaseEvaluatorTest, unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_datasets(self):
        for getter in get_dataset_getters():
            testname = '%s_%s' % (os.path.basename(__file__).
                                  replace('.pyc', '').replace('.py', ''),
                                  getter.__name__)

            with self.subTest(testname):
                backend_mock = get_evaluation_backend()
                D = getter()
                D_ = copy.deepcopy(D)
                y = D.data['Y_train']
                if len(y.shape) == 2 and y.shape[1] == 1:
                    D_.data['Y_train'] = y.flatten()
                backend_mock.load_datamanager.return_value = D_
                metric_lookup = {MULTILABEL_CLASSIFICATION: f1_macro,
                                 BINARY_CLASSIFICATION: accuracy,
                                 MULTICLASS_CLASSIFICATION: accuracy,
                                 REGRESSION: r2}
                queue_ = multiprocessing.Queue()

                evaluator = TestEvaluator(
                    backend_mock,
                    queue_,
                    metric=metric_lookup[D.info['task']],
                    port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                )

                evaluator.fit_predict_and_loss()
                rval = read_queue(evaluator.queue)
                self.assertEqual(len(rval), 1)
                self.assertEqual(len(rval[0]), 3)
                self.assertTrue(np.isfinite(rval[0]['loss']))


class FunctionsTest(unittest.TestCase):
    def setUp(self):
        self.queue = multiprocessing.Queue()
        self.configuration = get_configuration_space(
            {'task': MULTICLASS_CLASSIFICATION,
             'is_sparse': False}).get_default_configuration()
        self.data = get_multiclass_classification_datamanager()
        self.tmp_dir = os.path.join(os.path.dirname(__file__),
                                    '.test_cv_functions')
        self.backend = unittest.mock.Mock(spec=Backend)
        self.backend.temporary_directory = tempfile.gettempdir()
        self.backend.load_datamanager.return_value = self.data
        self.dataset_name = json.dumps({'task_id': 'test'})

        self.port = logging.handlers.DEFAULT_TCP_LOGGING_PORT

    def tearDown(self):
        try:
            shutil.rmtree(self.tmp_dir)
        except Exception:
            pass

    def test_eval_test(self):
        eval_t(
            queue=self.queue,
            backend=self.backend,
            config=self.configuration,
            metric=accuracy,
            seed=1, num_run=1,
            scoring_functions=None,
            output_y_hat_optimization=False,
            include=None,
            exclude=None,
            disable_file_output=False,
            instance=self.dataset_name,
            port=self.port,
        )
        rval = read_queue(self.queue)
        self.assertEqual(len(rval), 1)
        self.assertAlmostEqual(rval[0]['loss'], 0.040000000000000036)
        self.assertEqual(rval[0]['status'], StatusType.SUCCESS)
        self.assertNotIn('bac_metric', rval[0]['additional_run_info'])

    def test_eval_test_all_loss_functions(self):
        eval_t(
            queue=self.queue,
            backend=self.backend,
            config=self.configuration,
            metric=accuracy,
            seed=1, num_run=1,
            scoring_functions=SCORER_LIST,
            output_y_hat_optimization=False,
            include=None,
            exclude=None,
            disable_file_output=False,
            instance=self.dataset_name,
            port=self.port,
        )
        rval = read_queue(self.queue)
        self.assertEqual(len(rval), 1)

        # Note: All metric here should be minimized
        fixture = {'accuracy': 0.040000000000000036,
                   'balanced_accuracy': 0.02777777777777779,
                   'f1_macro': 0.0341005967604433,
                   'f1_micro': 0.040000000000000036,
                   'f1_weighted': 0.039693094629155934,
                   'log_loss': 0.148586485311389,
                   'precision_macro': 0.03703703703703709,
                   'precision_micro': 0.040000000000000036,
                   'precision_weighted': 0.03555555555555556,
                   'recall_macro': 0.02777777777777779,
                   'recall_micro': 0.040000000000000036,
                   'recall_weighted': 0.040000000000000036,
                   'num_run': -1}

        additional_run_info = rval[0]['additional_run_info']
        for key, value in fixture.items():
            self.assertAlmostEqual(additional_run_info[key], fixture[key], msg=key)
        self.assertEqual(len(additional_run_info), len(fixture) + 1,
                         msg=sorted(additional_run_info.items()))
        self.assertIn('duration', additional_run_info)
        self.assertAlmostEqual(rval[0]['loss'], 0.040000000000000036)
        self.assertEqual(rval[0]['status'], StatusType.SUCCESS)
