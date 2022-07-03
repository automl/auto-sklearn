from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse

# General TYPE definitions for numpy
TYPE_ADDITIONAL_INFO = Dict[str, Union[int, float, str, Dict, List, Tuple]]

FEAT_TYPE_TYPE = Dict[Union[str, int], str]
RANDOM_STATE_TYPE = Union[int, np.random.RandomState]
INIT_PARAMS_TYPE = Dict[str, Any]
DATASET_PROPERTIES_TYPE = Dict[str, Any]

INCLUDE_TYPE = Optional[List[str]]
EXCLUDE_TYPE = Optional[List[str]]

INCLUDE_BASE_TYPE = Dict[str, str]
EXCLUDE_BASE_TYPE = Dict[str, str]

INCLUDE_CLASSIFICATION_TYPE = Dict[str, List[str]]
EXCLUDE_CLASSIFICATION_TYPE = Dict[str, List[str]]

INCLUDE_REGRESSION_TYPE = Dict[str, List[str]]
EXCLUDE_REGRESSION_TYPE = Dict[str, List[str]]

INCLUDE_PIPELINE_TYPE = Dict[str, List[str]]
EXCLUDE_PIPELINE_TYPE = Dict[str, List[str]]

FIT_PARAMS_TYPE = Dict[str, Any]
SAMPLE_WEIGHT_TYPE = Union[np.ndarray, List]

PIPELINE_DATA_DTYPE = Union[
    np.ndarray,
    scipy.sparse.bsr_matrix,
    scipy.sparse.coo_matrix,
    scipy.sparse.csc_matrix,
    scipy.sparse.csr_matrix,
    scipy.sparse.dia_matrix,
    scipy.sparse.dok_matrix,
    scipy.sparse.lil_matrix,
]
