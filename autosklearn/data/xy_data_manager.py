# -*- encoding: utf-8 -*-
from typing import Dict, Optional, Union, cast

import numpy as np
import pandas as pd
from scipy import sparse

from autosklearn.constants import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
    MULTIOUTPUT_REGRESSION,
    REGRESSION,
)
from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.data.validation import SUPPORTED_FEAT_TYPES, SUPPORTED_TARGET_TYPES


class XYDataManager(AbstractDataManager):
    def __init__(
        self,
        X: SUPPORTED_FEAT_TYPES,
        y: SUPPORTED_TARGET_TYPES,
        X_test: Optional[SUPPORTED_FEAT_TYPES],
        y_test: Optional[SUPPORTED_TARGET_TYPES],
        task: int,
        feat_type: Dict[Union[str, int], str],
        dataset_name: str,
    ):
        super(XYDataManager, self).__init__(dataset_name)

        self.info["task"] = task
        if sparse.issparse(X):
            self.info["is_sparse"] = 1
            self.info["has_missing"] = np.all(
                np.isfinite(cast(sparse.csr_matrix, X).data)
            )
        else:
            self.info["is_sparse"] = 0
            if hasattr(X, "iloc"):
                self.info["has_missing"] = cast(pd.DataFrame, X).isnull().values.any()
            else:
                self.info["has_missing"] = np.all(np.isfinite(X))

        label_num = {
            REGRESSION: 1,
            BINARY_CLASSIFICATION: 2,
            MULTIOUTPUT_REGRESSION: np.shape(y)[-1],
            MULTICLASS_CLASSIFICATION: len(np.unique(y)),
            MULTILABEL_CLASSIFICATION: np.shape(y)[-1],
        }

        self.info["label_num"] = label_num[task]

        self.data["X_train"] = X
        self.data["Y_train"] = y
        if X_test is not None:
            self.data["X_test"] = X_test
        if y_test is not None:
            self.data["Y_test"] = y_test

        if isinstance(feat_type, dict):
            self.feat_type = feat_type
        else:
            raise ValueError(
                "Unsupported feat_type provided. We expect the user to "
                "provide a Dict[str, str] mapping from column to categorical/ "
                "numerical."
            )

        # TODO: try to guess task type!

        if len(np.shape(y)) > 2:
            raise ValueError(
                "y must not have more than two dimensions, "
                "but has %d." % len(np.shape(y))
            )

        if np.shape(X)[0] != np.shape(y)[0]:
            raise ValueError(
                "X and y must have the same number of "
                "datapoints, but have %d and %d." % (np.shape(X)[0], np.shape(y)[0])
            )
