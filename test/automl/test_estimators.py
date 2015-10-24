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
        automl = autosklearn.AutoSklearnClassifier(time_left_for_this_task=12,
                                                   per_run_time_limit=12)
        automl.fit(X_train, Y_train)
        score = automl.score(X_test, Y_test)
        print(automl.show_models())
        self.assertGreaterEqual(score, 0.8)
        self.assertEqual(automl._task, MULTICLASS_CLASSIFICATION)

        del automl
        self._tearDown(output)
