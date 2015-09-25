# -*- encoding: utf-8 -*-
import os

import numpy as np
import ParamSklearn.util as putil

import autosklearn
from . import Base


class EstimatorTest(Base):
    _multiprocess_can_split_ = True

    def test_fit_OneHotEncoder(self):
        output = os.path.join(self.test_dir, '..',
                              '.tmp_test_fit_OneHotEncoder')
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
        X_train = np.hstack((X_train,
                             np.arange(X_train.shape[0]).reshape((-1, 1))))
        cls = autosklearn.AutoSklearnClassifier(time_left_for_this_task=5,
                                                per_run_time_limit=5,
                                                output_folder=output,
                                                tmp_folder=output)
        cls.fit(X_train, Y_train,
                feat_type=['NUMERICAL', 'NUMERICAL', 'NUMERICAL', 'NUMERICAL',
                           'CATEGORICAL'])
        self.assertEqual([False, False, False, False, True],
                         cls._ohe.categorical_features)

        del cls
        self._tearDown(output)
