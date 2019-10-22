"""
=======
Holdout
=======

In *auto-sklearn* it is possible to use different resampling strategies
by specifying the arguments ``resampling_strategy`` and
``resampling_strategy_arguments``. The following example shows how to use the
holdout method as well as set the train-test split ratio when instantiating
``AutoSklearnClassifier``.
"""

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import numpy as np

import autosklearn.classification


def create_data_set_2(instances=1000, n_feats=10, n_categ_feats=4,
    categs_per_feat=5, missing_share=.3, n_classes=3):
    np.random.seed(0)
    
    # Create a base dataset
    size = (instances, n_feats)
    data = np.random.uniform(size=size)
    # Overwrite some features to make them categorical
    categ_flag = np.random.choice(n_feats, n_categ_feats, replace=False)
    numer_flag = list(set(range(n_feats)) - set(categ_flag))
    categ_data = np.random.randint(0, categs_per_feat, size=(instances, n_categ_feats))
    #data[:, categ_flag] = (categ_data * 42).astype(str)  # just to make it 'more non-numerical'
    data[:, categ_flag] = categ_data
    # Add missing values
    missing_mask = np.random.uniform(size=size) < missing_share
    data[missing_mask] = None
    # Create labels
    labels = np.random.randint(0, n_classes, size=instances)
    # Feature types
    feat_type = np.array(["categorical"] * n_feats)
    feat_type[numer_flag] = "numerical"

    return data, labels, feat_type.tolist()

def main():
    X, y, feat_type = create_data_set_2(n_categ_feats=4)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60,
        per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_holdout_example_tmp',
        output_folder='/tmp/autosklearn_holdout_example_out',
        disable_evaluator_output=False,
        # 'holdout' with 'train_size'=0.67 is the default argument setting
        # for AutoSklearnClassifier. It is explicitly specified in this example
        # for demonstrational purpose.
        resampling_strategy='holdout',
        resampling_strategy_arguments={'train_size': 0.67},
        #n_jobs=1
    )
    automl.fit(X_train, y_train, feat_type=feat_type, dataset_name='breast_cancer')

    # Print the final ensemble constructed by auto-sklearn.
    print(automl.show_models())
    predictions = automl.predict(X_test)
    # Print statistics about the auto-sklearn run such as number of
    # iterations, number of models failed with a time out.
    print(automl.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


if __name__ == '__main__':
    main()
