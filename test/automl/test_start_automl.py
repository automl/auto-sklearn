# -*- encoding: utf-8 -*-
import cPickle
import multiprocessing
import os
import shutil
import unittest

import numpy as np

import autosklearn.automl
import ParamSklearn.util as putil
from autosklearn.constants import *


class AutoMLTest(unittest.TestCase):

    def setUp(self):
        self.test_dir = os.path.dirname(__file__)
        self.output = os.path.join(self.test_dir, '..', '.tmp')
        try:
            shutil.rmtree(self.output)
        except Exception:
            pass
        os.makedirs(self.output)

    def tearDown(self):
        try:
            shutil.rmtree(self.output)
        except Exception as e:
            print(e)

    def test_fit(self):
        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
        automl = autosklearn.automl.AutoML(self.output, self.output, 10, 10)
        automl.fit(X_train, Y_train)
        score = automl.score(X_test, Y_test)
        self.assertGreaterEqual(score, 0.9)
        self.assertEqual(automl.task_, MULTICLASS_CLASSIFICATION)

    def test_dataset_manager_pickling(self):
        data_dir = os.path.join(self.test_dir, '..', '.data')
        dataset = '401_bac'
        data_manager_file = os.path.join(self.output,
                                         '%s_Manager.pkl' % dataset)

        queue = multiprocessing.Queue()
        auto = autosklearn.automl.AutoML(
            self.output, self.output, 10, 10,
            initial_configurations_via_metalearning=25,
            queue=queue)
        auto.fit_automl_dataset(dataset, data_dir)
        with open(data_manager_file) as fh:
            D = cPickle.load(fh)
            self.assertTrue(np.allclose(D.data['X_train'].data[:3], [1., 1.,
                                                                     2.]))

        time_needed_to_load_data, data_manager_file, proc_smac, proc_ensembles = \
            queue.get()
        proc_smac.wait()
        proc_ensembles.wait()
