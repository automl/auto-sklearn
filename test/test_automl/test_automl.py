# -*- encoding: utf-8 -*-
import multiprocessing
import os
import sys
import unittest
import unittest.mock

import numpy as np

from autosklearn.util.backend import Backend, BackendContext
from autosklearn.automl import AutoML

sys.path.append(os.path.dirname(__file__))
from base import Base


class AutoMLTest(Base, unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_refit_shuffle_on_fail(self):
        output = os.path.join(self.test_dir, '..', '.tmp_refit_shuffle_on_fail')
        context = BackendContext(output, output, False, False)
        backend = Backend(context)

        failing_model = unittest.mock.Mock()
        failing_model.fit.side_effect = [ValueError(), ValueError(), None]

        auto = AutoML(backend, 30, 30)
        ensemble_mock = unittest.mock.Mock()
        auto.ensemble_ = ensemble_mock
        ensemble_mock.get_model_identifiers.return_value = [1]

        auto.models_ = {1: failing_model}

        X = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        auto.refit(X, y)

        self.assertEqual(failing_model.fit.call_count, 3)


