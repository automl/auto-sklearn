from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import numpy as np
    from smac.optimizer.smbo import SMBO
    from smac.runhistory.runhistory import RunInfo, RunValue

    from autosklearn.automl import AutoMLClassifier


def test_early_stopping(
    make_automl_classifier: Callable[..., AutoMLClassifier],
    make_sklearn_dataset: Callable[..., tuple[np.ndarray, ...]],
) -> None:
    """
    Expects
    -------
    * Should early after fitting 2 models
    """

    def callback(
        smbo: SMBO,
        run_info: RunInfo,
        result: RunValue,
        time_left: float,
    ) -> bool | None:
        if int(result.additional_info["num_run"]) >= 2:
            return False

    automl = make_automl_classifier(get_trials_callback=callback)

    X_train, Y_train, X_test, Y_test = make_sklearn_dataset("iris")
    automl.fit(X_train, Y_train)

    assert len(automl.runhistory_.data) == 2
