"""See the contents of TOY_DATA for full details

/data
└── .auto-sklearn
    ├── runs
    │   ├── 0_1_0.0
    │   │   ├── 0.1.0.0.model
    │   │   ├── predictions_ensemble_0_1_0.0.npy
    │   │   ├── predictions_test_0_1_0.0.npy
    │   │   └── predictions_valid_0_1_0.0.npy
    │   ├── 0_2_0.0
    │   │   ├── 0.2.0.0.model
    │   │   ├── predictions_ensemble_0_2_0.0.npy
    │   │   ├── predictions_test_0_2_0.0.np
    │   │   ├── predictions_test_0_2_0.0.npy
    │   │   └── predictions_valid_0_2_0.0.npy
    │   └── 0_3_100.0
    │       ├── 0.3.0.0.model
    │       ├── 0.3.100.0.model
    │       ├── predictions_ensemble_0_3_100.0.npy
    │       ├── predictions_test_0_3_100.0.npy
    │       └── predictions_valid_0_3_100.0.npy
    ├── datamanager.pkl
    ├── true_targets_ensemble.npy
    └── predictions_ensemble_true.npy

# Ensemble targets and predictions
Both `predictions_ensemble_targets` and `true_targets_ensemble` are the same set of data
* [ [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], ]

# 0_1_0.0
All of run 0_1_0.0's predictions for "ensemble" "test" and "valid" are differing by
their predictions in the first key.
* [ [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], ]

# 0_2_0.0, 0_3_100.0
All of run 0_2_0.0's predictions for "ensemble" "test" and "valid" are exactly the same
as the `true_targets_ensemble` and `predictions_ensemble_true`
* [ [1, 0], [0, 1], [0, 1], [0, 1], [0, 1], ]

# Models
The models are empty files.

# Datamanager
The datamanager contains the iris dataset as the above numbers are made up with no
real corresponding models so the data from the datamanager can not be faked so easily.

# Extra Notes
The extra `predictions_test_0_2_0.0.np` are required to make `test_max_models_on_disc`
pass as it factors into the memory estimation. Should probably fix that.
"""

from typing import Callable

import pickle
from pathlib import Path

import numpy as np

from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.constants import BINARY_CLASSIFICATION
from autosklearn.data.xy_data_manager import XYDataManager

from pytest_cases import case

HERE = Path(__file__).parent.resolve()
DATADIR = HERE / "data"


@case
def case_3_models(
    tmp_path: Path,
    make_backend: Callable[..., Backend],
    make_sklearn_dataset: Callable[..., XYDataManager],
) -> Backend:
    """Gives the backend for the this certain setup"""
    path = tmp_path / "backend"

    # Create the datamanager that was used if needed
    dm_path = DATADIR / ".auto-sklearn" / "datamanager.pkl"

    if not dm_path.exists():
        datamanager = make_sklearn_dataset(
            name="breast_cancer",
            task=BINARY_CLASSIFICATION,
            feat_type="numerical",  # They're all numerical
            as_datamanager=True,
        )

        # For some reason, the old mock was just returning this array as:
        #
        #   datamanger.data.get.return_value = array
        #
        model_3_path = DATADIR / ".auto-sklearn" / "runs" / "0_3_100.0"
        test_preds = model_3_path / "predictions_test_0_3_100.0.npy"
        array = np.load(test_preds)

        datamanager.data["Y_valid"] = array
        datamanager.data["Y_test"] = array

        with dm_path.open("wb") as f:
            pickle.dump(datamanager, f)

    return make_backend(path=path, template=DATADIR)
