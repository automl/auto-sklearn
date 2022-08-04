import numpy as np

from autosklearn.constants import REGRESSION
from autosklearn.ensembles.singlebest_ensemble import SingleBest, SingleModelEnsemble
from autosklearn.metrics import root_mean_squared_error


def test_SingleBest(backend):

    ensemble = SingleBest(
        task_type=REGRESSION,
        metrics=[root_mean_squared_error],
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

    # Check that the weight isn't touched
    assert ensemble.weights_ == [1.0]

    ensemble.fit(
        base_models_predictions=predictions,
        X_data=X_data,
        true_targets=y_true,
        model_identifiers=identifiers,
        runs=[],
    )

    assert ensemble.weights_ == [1.0]
    assert ensemble.indices_ == [14]
    assert ensemble.identifiers_ == [(14, 14, 0.0)]
    assert ensemble.best_model_score_ == 0.5

    # test the prediction
    ensemble_prediction = np.random.random((1, 100, 2))
    prediction = ensemble.predict(ensemble_prediction)
    assert prediction.shape == (100, 2)

    # test string representation; it selects the 14th prediction (6)
    # and ties with the 15th (5), but then afterwards the RMSE goes up again
    assert (
        str(ensemble)
        == """SingleBest:
\tMembers: [14]
\tWeights: [1.0]
\tIdentifiers: [(14, 14, 0.0)]"""
    )

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


def test_SingleModelEnsemble(backend):

    ensemble = SingleModelEnsemble(
        task_type=REGRESSION,
        metrics=[root_mean_squared_error],
        random_state=0,
        backend=backend,
        model_index=5,
    )

    # fit the whole thing
    X_data = np.random.random(size=(100, 2))
    y_true = np.full((100), 5.5)
    predictions = []
    for i in range(20, 1, -1):
        pred = np.ones((100), dtype=np.float32) * i
        predictions.append(pred)
    identifiers = [(i, i, 0.0) for i in range(20)]

    # Check that the weight isn't touched
    assert ensemble.weights_ == [1.0]

    ensemble.fit(
        base_models_predictions=predictions,
        X_data=X_data,
        true_targets=y_true,
        model_identifiers=identifiers,
        runs=[],
    )

    assert ensemble.weights_ == [1.0]
    assert ensemble.indices_ == [5]
    assert ensemble.identifiers_ == [(5, 5, 0.0)]
    assert ensemble.best_model_score_ == 9.5

    # test the prediction
    ensemble_prediction = np.random.random((1, 100, 2))
    prediction = ensemble.predict(ensemble_prediction)
    assert prediction.shape == (100, 2)

    assert (
        str(ensemble)
        == """SingleModelEnsemble:
\tMembers: [5]
\tWeights: [1.0]
\tIdentifiers: [(5, 5, 0.0)]"""
    )

    models_with_weights = ensemble.get_models_with_weights(
        {identifier: i for i, identifier in enumerate(identifiers)}
    )
    assert models_with_weights == [(1.0, 5)]

    identifiers_with_weights = ensemble.get_identifiers_with_weights()
    assert identifiers_with_weights == [((5, 5, 0.0), 1.0)]

    selected_model_identifiers = ensemble.get_selected_model_identifiers()
    assert selected_model_identifiers == [(5, 5, 0.0)]

    best_model_score = ensemble.get_validation_performance()
    assert best_model_score == 9.5
