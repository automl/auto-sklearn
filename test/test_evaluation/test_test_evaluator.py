# -*- encoding: utf-8 -*-
from __future__ import print_function
import copy
import multiprocessing
import os
import shutil
import sys
import unittest
import unittest.mock

import numpy as np

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import get_dataset_getters, BaseEvaluatorTest, \
    get_multiclass_classification_datamanager
from autosklearn.constants import *
from autosklearn.evaluation import TestEvaluator
# Otherwise nosetests thinks this is a test to run...
from autosklearn.evaluation import eval_t, get_last_result
from autosklearn.util.pipeline import get_configuration_space
from autosklearn.util import Backend

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
                backend_mock = unittest.mock.Mock(spec=Backend)
                backend_mock.get_model_dir.return_value = 'dutirapbdxvltcrpbdlcatepdeau'
                D = getter()
                D_ = copy.deepcopy(D)
                y = D.data['Y_train']
                if len(y.shape) == 2 and y.shape[1] == 1:
                    y = y.flatten()
                queue_ = multiprocessing.Queue()
                evaluator = TestEvaluator(D_, backend_mock, queue_)

                evaluator.fit_predict_and_loss()
                duration, result, seed, run_info, status = evaluator.queue.get(timeout=1)
                self.assertTrue(np.isfinite(result))


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

    def tearDown(self):
        try:
            shutil.rmtree(self.tmp_dir)
        except Exception:
            pass

    def test_eval_test(self):
        eval_t(queue=self.queue,
               backend=self.backend,
               config=self.configuration,
               data=self.data,
               seed=1, num_run=1, subsample=None, with_predictions=True,
               all_scoring_functions=False, output_y_test=True,
               include=None, exclude=None, disable_file_output=False)
        info = get_last_result(self.queue)
        self.assertAlmostEqual(info[1], 0.041666666666666852)
        self.assertEqual(info[2], 1)
        self.assertNotIn('bac_metric', info[3])

    def test_eval_test_all_loss_functions(self):
        eval_t(queue=self.queue,
               backend=self.backend,
               config=self.configuration,
               data=self.data,
               seed=1, num_run=1, subsample=None, with_predictions=True,
               all_scoring_functions=True, output_y_test=True,
               include=None, exclude=None, disable_file_output=False)
        info = get_last_result(self.queue)
        self.assertIn(
            'f1_metric: 0.0511508951407;pac_metric: 0.185257565321;'
            'acc_metric: 0.06;auc_metric: 0.00917546505782;'
            'bac_metric: 0.0416666666667;duration: ', info[3])
        self.assertAlmostEqual(info[1], 0.041666666666666852)
        self.assertEqual(info[2], 1)
