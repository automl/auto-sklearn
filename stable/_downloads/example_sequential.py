"""
================
Sequential Usage
================

By default, *auto-sklearn* fits the machine learning models and build their
ensembles in parallel. However, it is also possible to run the two processes
sequentially. The example below shows how to first fit the models and build the
ensembles afterwards.
"""

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification


def main():
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_sequential_example_tmp',
        output_folder='/tmp/autosklearn_sequential_example_out',
        # Do not construct ensembles in parallel to avoid using more than one
        # core at a time. The ensemble will be constructed after auto-sklearn
        # finished fitting all machine learning models.
        ensemble_size=0,
        delete_tmp_folder_after_terminate=False,
    )
    automl.fit(X_train, y_train, dataset_name='digits')
    # This call to fit_ensemble uses all models trained in the previous call
    # to fit to build an ensemble which can be used with automl.predict()
    automl.fit_ensemble(y_train, ensemble_size=50)

    print(automl.show_models())
    predictions = automl.predict(X_test)
    print(automl.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


if __name__ == '__main__':
    main()
