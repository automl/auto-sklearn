import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification


def main():
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120, per_run_time_limit=30,
        tmp_folder='/tmp/autoslearn_holdout_example_tmp',
        output_folder='/tmp/autosklearn_holdout_example_out',
        disable_evaluator_output=False)
    automl.fit(X_train, y_train, dataset_name='digits')

    # Print the final ensemble constructed by auto-sklearn.
    print(automl.show_models())
    predictions = automl.predict(X_test)
    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


if __name__ == '__main__':
    main()
