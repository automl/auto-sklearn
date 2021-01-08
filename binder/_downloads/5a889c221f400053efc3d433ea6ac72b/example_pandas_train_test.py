# -*- encoding: utf-8 -*-
"""
===============================
Test and Train data with Pandas
===============================

*auto-sklearn* can automatically encode categorical columns using a label/ordinal encoder.
This example highlights how to properly set the dtype in a DataFrame for this to happen,
and showcase how to input also testing data to autosklearn.
The X_train/y_train arguments to the fit function will be used to fit the scikit-learn model,
whereas the X_test/y_test will be used to evaluate how good this scikit-learn model generalizes
to unseen data (i.e. data not in X_train/y_train). Using test data is a good mechanism to measure
if the trained model suffers from overfit, and more details can be found on `evaluating estimator
performance <https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation>`_.
This example further highlights through a plot, the best individual models found by *auto-sklearn*
through time (under single_best_optimization_score/single_best_test_score's legend).
It also shows the training and test performance of the ensemble build using the best
performing models (under ensemble_optimization_score and ensemble_test_score respectively).

There is also support to manually indicate the feature types (whether a column is categorical
or numerical) via the argument feat_types from fit(). This is important when working with
list or numpy arrays as there is no per-column dtype (further details in the example
`Continuous and categorical data <example_feature_types.html>`_).
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from smac.tae import StatusType

import autosklearn.classification


def get_runhistory_models_performance(automl):
    metric = cls.automl_._metric
    data = automl.automl_.runhistory_.data
    performance_list = []
    for run_key, run_value in data.items():
        if run_value.status != StatusType.SUCCESS:
            # Ignore crashed runs
            continue
        # Alternatively, it is possible to also obtain the start time with ``run_value.starttime``
        endtime = pd.Timestamp(time.strftime('%Y-%m-%d %H:%M:%S',
                                             time.localtime(run_value.endtime)))
        val_score = metric._optimum - (metric._sign * run_value.cost)
        test_score = metric._optimum - (metric._sign * run_value.additional_info['test_loss'])
        train_score = metric._optimum - (metric._sign * run_value.additional_info['train_loss'])
        performance_list.append({
            'Timestamp': endtime,
            'single_best_optimization_score': val_score,
            'single_best_test_score': test_score,
            'single_best_train_score': train_score,
        })
    return pd.DataFrame(performance_list)


if __name__ == "__main__":
    ############################################################################
    # Data Loading
    # ============

    # Using Australian dataset https://www.openml.org/d/40981.
    # This example will use the command fetch_openml, which will
    # download a properly formatted dataframe if you use as_frame=True.
    # For demonstration purposes, we will download a numpy array using
    # as_frame=False, and manually creating the pandas DataFrame
    X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True, as_frame=False)

    # bool and category will be automatically encoded.
    # Targets for classification are also automatically encoded
    # If using fetch_openml, data is already properly encoded, below
    # is an example for user reference
    X = pd.DataFrame(
        data=X,
        columns=['A' + str(i) for i in range(1, 15)]
    )
    desired_boolean_columns = ['A1']
    desired_categorical_columns = ['A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12']
    desired_numerical_columns = ['A2', 'A3', 'A7', 'A10', 'A13', 'A14']
    for column in X.columns:
        if column in desired_boolean_columns:
            X[column] = X[column].astype('bool')
        elif column in desired_categorical_columns:
            X[column] = X[column].astype('category')
        else:
            X[column] = pd.to_numeric(X[column])

    y = pd.DataFrame(y, dtype='category')

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.5, random_state=3
    )
    print(X.dtypes)

    ############################################################################
    # Build and fit a classifier
    # ==========================

    cls = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
    )
    cls.fit(X_train, y_train, X_test, y_test)

    ###########################################################################
    # Get the Score of the final ensemble
    # ===================================

    predictions = cls.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))

    ############################################################################
    # Plot the ensemble performance
    # ===================================

    ensemble_performance_frame = pd.DataFrame(cls.automl_.ensemble_performance_history)
    best_values = pd.Series({'ensemble_optimization_score': -np.inf,
                             'ensemble_test_score': -np.inf})
    for idx in ensemble_performance_frame.index:
        if (
            ensemble_performance_frame.loc[idx, 'ensemble_optimization_score']
            > best_values['ensemble_optimization_score']
        ):
            best_values = ensemble_performance_frame.loc[idx]
        ensemble_performance_frame.loc[idx] = best_values

    individual_performance_frame = get_runhistory_models_performance(cls)
    best_values = pd.Series({'single_best_optimization_score': -np.inf,
                             'single_best_test_score': -np.inf,
                             'single_best_train_score': -np.inf})
    for idx in individual_performance_frame.index:
        if (
            individual_performance_frame.loc[idx, 'single_best_optimization_score']
            > best_values['single_best_optimization_score']
        ):
            best_values = individual_performance_frame.loc[idx]
        individual_performance_frame.loc[idx] = best_values

    pd.merge(
        ensemble_performance_frame,
        individual_performance_frame,
        on="Timestamp", how='outer'
    ).sort_values('Timestamp').fillna(method='ffill').plot(
        x='Timestamp',
        kind='line',
        legend=True,
        title='Auto-sklearn accuracy over time',
        grid=True,
    )
    plt.show()
