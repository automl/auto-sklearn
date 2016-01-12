# -*- encoding: utf-8 -*-
from __future__ import print_function

import multiprocessing
import os
import sys
import time

import numpy as np
import six
import sklearn.datasets

import autosklearn.automl
import autosklearn.pipeline.util as putil
from autosklearn.util import setup_logger, get_logger
from autosklearn.constants import *
from autosklearn.smbo import load_data

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

        output = os.path.join(self.test_dir, '..', '.tmp_test_binary_score')
        self._setUp(output)

        data = sklearn.datasets.make_classification(
            n_samples=1000, n_features=20, n_redundant=5, n_informative=5,
            n_repeated=2, n_clusters_per_class=2, random_state=1)
        X_train = data[0][:700]
        Y_train = data[1][:700]
        X_test = data[0][700:]
        Y_test = data[1][700:]

        automl = autosklearn.automl.AutoML(output, output, 15, 15)
        automl.fit(X_train, Y_train, task=BINARY_CLASSIFICATION)
        self.assertEqual(automl._task, BINARY_CLASSIFICATION)

        score = automl.score(X_test, Y_test)
        self.assertGreaterEqual(score, 0.5)

        del automl
        self._tearDown(output)

    def test_automl_outputs(self):
        output = os.path.join(self.test_dir, '..',
                              '.tmp_test_automl_outputs')
        self._setUp(output)
        name = '31_bac'
        dataset = os.path.join(self.test_dir, '..', '.data', name)
        data_manager_file = os.path.join(output, '.auto-sklearn',
                                         'datamanager.pkl')

        queue = multiprocessing.Queue()
        auto = autosklearn.automl.AutoML(
            output, output, 15, 15,
            initial_configurations_via_metalearning=25,
            queue=queue,
            seed=100)
        auto.fit_automl_dataset(dataset)

        # pickled data manager (without one hot encoding!)
        with open(data_manager_file, 'rb') as fh:
            D = six.moves.cPickle.load(fh)
            self.assertTrue(np.allclose(D.data['X_train'][0, :3],
                                        [1., 12., 2.]))

        time_needed_to_load_data, data_manager_file, procs = \
            queue.get()
        for proc in procs:
            proc.wait()

        # Start time
        print(os.listdir(os.path.join(output, '.auto-sklearn')))
        start_time_file_path = os.path.join(output, '.auto-sklearn',
                                            "start_time_100")
        with open(start_time_file_path, 'r') as fh:
            start_time = float(fh.read())
        self.assertGreaterEqual(time.time() - start_time, 10)

        del auto
        self._tearDown(output)

    def test_do_dummy_prediction(self):
        for name in ['401_bac', '31_bac', 'adult', 'cadata']:
            output = os.path.join(self.test_dir, '..',
                                  '.tmp_test_do_dummy_prediction')
            self._setUp(output)

            dataset = os.path.join(self.test_dir, '..', '.data', name)

            auto = autosklearn.automl.AutoML(
                output, output, 15, 15,
                initial_configurations_via_metalearning=25)
            setup_logger()
            auto._logger = get_logger('test_do_dummy_predictions')
            auto._backend._make_internals_directory()
            D = load_data(dataset, output)
            auto._backend.save_datamanager(D)
            auto._do_dummy_prediction(D)

            # Assure that the dummy predictions are not in the current working
            # directory, but in the output directory (under output)
            self.assertFalse(os.path.exists(os.path.join(os.getcwd(),
                                                         '.auto-sklearn')))
            self.assertTrue(os.path.exists(os.path.join(output,
                                                        '.auto-sklearn')))

            del auto
            self._tearDown(output)
