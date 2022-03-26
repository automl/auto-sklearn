from typing import Callable

import pickle
from pathlib import Path

import numpy as np

from autosklearn.automl_common.common.utils.backend import Backend
from autosklearn.constants import BINARY_CLASSIFICATION
from autosklearn.data.xy_data_manager import XYDataManager

from pytest_cases import case

HERE = Path(__file__).parent.resolve()


@case(tags=["backend", "setup_3_models"])
def case_backend_setup_3_models(
    tmp_path: Path,
    make_backend: Callable[..., Backend],
    make_sklearn_dataset: Callable[..., XYDataManager],
) -> Backend:
    """See the contents of TOY_DATA for full details

    /toy_data
        /.auto-sklearn
            /runs
                /0_1_0.0
                /0_2_0.0
                /0_3_100.0
            /datamanger.pkl
            /predictions_ensemble_targets.npy
            /true_targets_ensemble.npy  # Same file as predictions_ensemble_targets
    """
    path = tmp_path / "backend"
    TOY_DATA = HERE / "toy_data"

    # Create the datamanager that was used if needed
    dm_path = TOY_DATA / ".auto-sklearn" / "datamanager.pkl"

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
        model_3_path = TOY_DATA / ".auto-sklearn" / "runs" / "0_3_100.0"
        test_preds = model_3_path / "predictions_test_0_3_100.0.npy"
        array = np.load(test_preds)

        datamanager.data["Y_valid"] = array
        datamanager.data["Y_test"] = array

        with dm_path.open("wb") as f:
            pickle.dump(datamanager, f)

    return make_backend(path=path, template=TOY_DATA)
