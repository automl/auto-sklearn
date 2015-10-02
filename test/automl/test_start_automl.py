# -*- encoding: utf-8 -*-
from __future__ import print_function

import multiprocessing
import os
import time

import mock
import numpy as np
import six

import autosklearn.automl
from autosklearn.util import Backend
import ParamSklearn.util as putil
from autosklearn.constants import *
from autosklearn.cli.base_interface import store_and_or_load_data

from base import Base

class AutoMLTest(Base):
    _multiprocess_can_split_ = True

    def test_fit(self):
        output = os.path.join(self.test_dir, '..', '.tmp_test_fit')
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
        automl = autosklearn.automl.AutoML(output, output, 10, 10)
        automl.fit(X_train, Y_train)
        score = automl.score(X_test, Y_test)
        self.assertGreaterEqual(score, 0.9)
        self.assertEqual(automl._task, MULTICLASS_CLASSIFICATION)

        del automl
        self._tearDown(output)

    def test_automl_outputs(self):
        output = os.path.join(self.test_dir, '..',
                              '.tmp_test_automl_outputs')
        self._setUp(output)

        name = '401_bac'
        dataset = os.path.join(self.test_dir, '..', '.data', name)
        data_manager_file = os.path.join(output, '.auto-sklearn',
                                         'datamanager.pkl')

        queue = multiprocessing.Queue()
        auto = autosklearn.automl.AutoML(
            output, output, 10, 10,
            initial_configurations_via_metalearning=25,
            queue=queue)
        auto.fit_automl_dataset(dataset)

        # pickled data manager (with one hot encoding!)
        with open(data_manager_file) as fh:
            D = six.moves.cPickle.load(fh)
            self.assertTrue(np.allclose(D.data['X_train'].data[:3],
                                        [1., 1., 2.]))

        time_needed_to_load_data, data_manager_file, procs = \
            queue.get()
        for proc in procs:
            proc.wait()

        # Start time
        start_time_file_path = os.path.join(output, '.auto-sklearn',
                                            "start_time.txt")
        with open(start_time_file_path, 'r') as fh:
            start_time = float(fh.read())
        self.assertGreaterEqual(time.time() - start_time, 10)

        del auto
        self._tearDown(output)

    def test_do_dummy_prediction(self):
        output = os.path.join(self.test_dir, '..',
                              '.tmp_test_do_dummy_prediction')
        self._setUp(output)

        name = '401_bac'
        dataset = os.path.join(self.test_dir, '..', '.data', name)

        auto = autosklearn.automl.AutoML(
            output, output, 10, 10,
            initial_configurations_via_metalearning=25)
        auto._backend._make_internals_directory()
        D = store_and_or_load_data(dataset, output)
        auto._do_dummy_prediction(D)

        del auto
        self._tearDown(output)
