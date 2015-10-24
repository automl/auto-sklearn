# -*- encoding: utf-8 -*-
import os

import ParamSklearn.util as putil

import autosklearn
from autosklearn.constants import *
from base import Base


class EstimatorTest(Base):
    _multiprocess_can_split_ = True

    def test_fit(self):
        output = os.path.join(self.test_dir, '..', '.tmp_estimator_fit')
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
        automl = autosklearn.automl.AutoML(output, output, 12, 12)
        automl.fit(X_train, Y_train)
        score = automl.score(X_test, Y_test)
        self.assertGreaterEqual(score, 0.9)
        self.assertEqual(automl._task, MULTICLASS_CLASSIFICATION)

        del automl
        self._tearDown(output)
