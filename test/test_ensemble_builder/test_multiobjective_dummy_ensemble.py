import numpy as np
import sklearn.metrics

from autosklearn.constants import REGRESSION
from autosklearn.ensembles.multiobjective_dummy_ensemble import (
    MultiObjectiveDummyEnsemble,
)
from autosklearn.ensembles.singlebest_ensemble import SingleModelEnsemble
from autosklearn.metrics import MAXINT, make_scorer, root_mean_squared_error


def test_MultiObjectiveDummyEnsemble(backend):
    negative_root_mean_squared_error = make_scorer(
        "negative_root_mean_squared_error",
        sklearn.metrics.mean_squared_error,
        optimum=0,
        worst_possible_result=-MAXINT,
        greater_is_better=True,
        squared=False,
    )

    ensemble = MultiObjectiveDummyEnsemble(
        task_type=REGRESSION,
        metrics=[root_mean_squared_error, negative_root_mean_squared_error],
        random_state=0,
        backend=backend,
    )

    # fit the whole thing
    X_data = np.random.random(size=(100, 2))
    y_true = np.full((100), 5.5)
    predictions = []
    for i in range(20, 1, -1):
        pred = np.ones((100), dtype=np.float32) * i
        predictions.append(pred)
    identifiers = [(i, i, 0.0) for i in range(20)]

    ensemble.fit(
        base_models_predictions=predictions,
        X_data=X_data,
        true_targets=y_true,
        model_identifiers=identifiers,
        runs=[],
    )

    # Because the target value is 5.5 some of the predictions have the same error
    # -> not every prediction is on the Pareto front
    assert len(ensemble.pareto_set_) == 15
    for sub_ensemble in ensemble.pareto_set_:
        assert isinstance(sub_ensemble, SingleModelEnsemble)

    # test the prediction
    ensemble_prediction = np.random.random((1, 100, 2))
    prediction = ensemble.predict(ensemble_prediction)
    assert prediction.shape == (100, 2)

    assert str(ensemble) == """MultiObjectiveDummyEnsemble: 15 models"""

    # From here on everything is supposed to return the output of the best
    # ensemble according to the 1st metric
    models_with_weights = ensemble.get_models_with_weights(
        {identifier: i for i, identifier in enumerate(identifiers)}
    )
    assert models_with_weights == [(1.0, 14)]

    identifiers_with_weights = ensemble.get_identifiers_with_weights()
    assert identifiers_with_weights == [((14, 14, 0.0), 1.0)]

    selected_model_identifiers = ensemble.get_selected_model_identifiers()
    assert selected_model_identifiers == [(14, 14, 0.0)]

    best_model_score = ensemble.get_validation_performance()
    assert best_model_score == 0.5
