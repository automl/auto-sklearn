# -*- encoding: utf-8 -*-
from __future__ import print_function
import copy
import os
import shutil
import sys
import numpy as np

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import get_dataset_getters, BaseEvaluatorTest
from autosklearn.evaluation import TestEvaluator

N_TEST_RUNS = 10


class Dummy(object):
    pass


class TestEvaluator_Test(BaseEvaluatorTest):
    _multiprocess_can_split_ = True

    def teardown(self):
        try:
            shutil.rmtree(self.output_dir)
        except Exception:
            pass

        for output_dir in self.output_directories:
            try:
                shutil.rmtree(output_dir)
            except Exception:
                pass

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
                    evaluator = TestEvaluator(D_, output_directory, None)

                    err[i] = evaluator.fit_predict_and_loss()[0]

                    self.assertTrue(np.isfinite(err[i]))

