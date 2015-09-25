# -*- encoding: utf-8 -*-
from __future__ import print_function

import multiprocessing
import os
import time

import numpy as np
import six

import autosklearn.automl
import ParamSklearn.util as putil
from autosklearn.constants import *

from . import Base

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

    def test_dataset_manager_pickling(self):
        output = os.path.join(self.test_dir, '..',
                              '.tmp_test_dataset_manager_pickling')
        self._setUp(output)

        name = '401_bac'
        dataset = os.path.join(self.test_dir, '..', '.data', name)
        data_manager_file = os.path.join(output, '%s_Manager.pkl' % name)

        queue = multiprocessing.Queue()
        auto = autosklearn.automl.AutoML(
            output, output, 10, 10,
            initial_configurations_via_metalearning=25,
            queue=queue)
        auto.fit_automl_dataset(dataset)
        with open(data_manager_file) as fh:
            D = six.moves.cPickle.load(fh)
            self.assertTrue(np.allclose(D.data['X_train'].data[:3],
                                        [1., 1., 2.]))

        time_needed_to_load_data, data_manager_file, proc_smac, proc_ensembles = \
            queue.get()
        proc_smac.wait()
        proc_ensembles.wait()

        del auto
        self._tearDown(output)
