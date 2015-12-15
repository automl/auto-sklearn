# -*- encoding: utf-8 -*-
from __future__ import print_function

import multiprocessing
import os
import sys
import time

import numpy as np
import six

import autosklearn.automl
import ParamSklearn.util as putil
from autosklearn.constants import *
from autosklearn.cli.base_interface import store_and_or_load_data

sys.path.append(os.path.dirname(__file__))
from base import Base

class AutoMLTest(Base):
    _multiprocess_can_split_ = True

    def test_fit(self):
        if self.travis:
            self.skipTest('This test does currently not run on travis-ci. '
                          'Make sure it runs locally on your machine!')

        output = os.path.join(self.test_dir, '..', '.tmp_test_fit')
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
        automl = autosklearn.automl.AutoML(output, output, 15, 15)
        automl.fit(X_train, Y_train)
        score = automl.score(X_test, Y_test)
        self.assertGreaterEqual(score, 0.8)
        self.assertEqual(automl._task, MULTICLASS_CLASSIFICATION)

        del automl
        self._tearDown(output)

    def test_binary_score(self):
        """
        Test fix for binary classification prediction
        taking the index 1 of second dimension in prediction matrix
        """
        if self.travis:
            self.skipTest('This test does currently not run on travis-ci. '
                          'Make sure it runs locally on your machine!')

        output = os.path.join(self.test_dir, '..', '.tmp_test_fit')
        self._setUp(output)

        # Had to use this dummy dataset because
        # I cannot find a way to efficiently load a binary dataset
        # without changing files in paramsklearn or automl class

        X_train = np.random.rand(100, 20)
        Y_train = np.random.randint(0, 2, 100)

        automl = autosklearn.automl.AutoML(output, output, 30, 15)
        automl.fit(X_train, Y_train, task=BINARY_CLASSIFICATION)
        self.assertEqual(automl._task, BINARY_CLASSIFICATION)

        X_test = np.random.rand(50, 20)
        Y_test = np.random.randint(0, 2, 50)

        score = automl.score(X_test, Y_test)
        self.assertGreaterEqual(score, 0.0)

        del automl
        self._tearDown(output)

    def test_automl_outputs(self):
        print("1.")
        output = os.path.join(self.test_dir, '..',
                              '.tmp_test_automl_outputs')
        print("2.")
        self._setUp(output)
        print("3.")
        name = '31_bac'
        print("4.")
        dataset = os.path.join(self.test_dir, '..', '.data', name)
        print("5.")
        data_manager_file = os.path.join(output, '.auto-sklearn',
                                         'datamanager.pkl')
        print("6.")

        queue = multiprocessing.Queue()
        print("7.")
        auto = autosklearn.automl.AutoML(
            output, output, 15, 15,
            initial_configurations_via_metalearning=25,
            queue=queue,
            seed=100)
        print("8.")
        auto.fit_automl_dataset(dataset)
        print("9.")

        # pickled data manager (without one hot encoding!)
        with open(data_manager_file, 'rb') as fh:
            print("10.")
            D = six.moves.cPickle.load(fh)
            print("11.")
            self.assertTrue(np.allclose(D.data['X_train'][0, :3],
                                        [1., 12., 2.]))
            print("12.")

        print("13.")
        time_needed_to_load_data, data_manager_file, procs = \
            queue.get()
        print("14.")
        for proc in procs:
            proc.wait()
        print("15.")

        # Start time
        print(os.listdir(os.path.join(output, '.auto-sklearn')))
        print("16.")
        start_time_file_path = os.path.join(output, '.auto-sklearn',
                                            "start_time_100")
        print("17.")
        with open(start_time_file_path, 'r') as fh:
            print("18.")
            start_time = float(fh.read())
            print("19.")
        self.assertGreaterEqual(time.time() - start_time, 10)
        print("20.")

        del auto
        print("21.")
        self._tearDown(output)
        print("22.")

    def test_do_dummy_prediction(self):
        output = os.path.join(self.test_dir, '..',
                              '.tmp_test_do_dummy_prediction')
        self._setUp(output)

        name = '401_bac'
        dataset = os.path.join(self.test_dir, '..', '.data', name)

        auto = autosklearn.automl.AutoML(
            output, output, 15, 15,
            initial_configurations_via_metalearning=25)
        auto._backend._make_internals_directory()
        D = store_and_or_load_data(dataset, output)
        auto._do_dummy_prediction(D)

        del auto
        self._tearDown(output)
