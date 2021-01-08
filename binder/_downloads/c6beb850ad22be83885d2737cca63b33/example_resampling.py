"""
=====================
Resampling Strategies
=====================

In *auto-sklearn* it is possible to use different resampling strategies
by specifying the arguments ``resampling_strategy`` and
``resampling_strategy_arguments``. The following example shows common
settings for the ``AutoSklearnClassifier``.
"""

import numpy as np
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification


if __name__ == "__main__":
    ############################################################################
    # Data Loading
    # ============

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    ############################################################################
    # Holdout
    # =======

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_resampling_example_tmp',
        output_folder='/tmp/autosklearn_resampling_example_out',
        disable_evaluator_output=False,
        # 'holdout' with 'train_size'=0.67 is the default argument setting
        # for AutoSklearnClassifier. It is explicitly specified in this example
        # for demonstrational purpose.
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.67},
    )
    automl.fit(X_train, y_train, dataset_name='breast_cancer')

    ############################################################################
    # Get the Score of the final ensemble
    # ===================================

    predictions = automl.predict(X_test)
    print("Accuracy score holdout: ", sklearn.metrics.accuracy_score(y_test, predictions))


    ############################################################################
    # Cross-validation
    # ================

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_resampling_example_tmp',
        output_folder='/tmp/autosklearn_resampling_example_out',
        disable_evaluator_output=False,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5},
    )
    automl.fit(X_train, y_train, dataset_name='breast_cancer')

    # One can use models trained during cross-validation directly to predict
    # for unseen data. For this, all k models trained during k-fold
    # cross-validation are considered as a single soft-voting ensemble inside
    # the ensemble constructed with ensemble selection.
    print('Before re-fit')
    predictions = automl.predict(X_test)
    print("Accuracy score CV", sklearn.metrics.accuracy_score(y_test, predictions))

    ############################################################################
    # Perform a refit
    # ===============
    # During fit(), models are fit on individual cross-validation folds. To use
    # all available data, we call refit() which trains all models in the
    # final ensemble on the whole dataset.
    print('After re-fit')
    automl.refit(X_train.copy(), y_train.copy())
    predictions = automl.predict(X_test)
    print("Accuracy score CV", sklearn.metrics.accuracy_score(y_test, predictions))

    ############################################################################
    # scikit-learn splitter objects
    # =============================
    # It is also possible to use
    # `scikit-learn's splitter classes <https://scikit-learn.org/stable/modules/classes.html#splitter
    # -classes>`_ to further customize the outputs. In case one needs to have 100% control over the
    # splitting, it is possible to use
    # `scikit-learn's PredefinedSplit <https://scikit-learn.org/stable/modules/generated/
    # sklearn.model_selection.PredefinedSplit.html>`_.

    ############################################################################
    # Below is an example of using a predefined split. We split the training
    # data by the first feature. In practice, one would use a splitting according
    # to the use case at hand.

    resampling_strategy = sklearn.model_selection.PredefinedSplit
    resampling_strategy_arguments = {'test_fold': np.where(X_train[:, 0] < np.mean(X_train[:, 0]))[0]}

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_resampling_example_tmp',
        output_folder='/tmp/autosklearn_resampling_example_out',
        disable_evaluator_output=False,
        resampling_strategy=resampling_strategy,
        resampling_strategy_arguments=resampling_strategy_arguments,
    )
    automl.fit(X_train, y_train, dataset_name='breast_cancer')

    ############################################################################
    # For custom resampling strategies (i.e. resampling strategies that are not
    # defined as strings by Auto-sklearn) it is necessary to perform a refit:
    automl.refit(X_train, y_train)

    ############################################################################
    # Get the Score of the final ensemble (again)
    # ===========================================
    #
    # Obviously, this score is pretty bad as we "destroyed" the dataset by
    # splitting it on the first feature.
    predictions = automl.predict(X_test)
    print("Accuracy score custom split", sklearn.metrics.accuracy_score(y_test, predictions))
