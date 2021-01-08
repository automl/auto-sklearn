# -*- encoding: utf-8 -*-
"""
=============
Feature Types
=============

In *auto-sklearn* it is possible to specify the feature types of a dataset when calling the method
:meth:`fit() <autosklearn.classification.AutoSklearnClassifier.fit>` by specifying the argument
``feat_type``. The following example demonstrates a way it can be done.

Additionally, you can provide a properly formatted pandas DataFrame, and the feature
types will be automatically inferred, as demonstrated in
`Pandas Train and Test inputs <example_pandas_train_test.html>`_.
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
    # Load Australian dataset from https://www.openml.org/d/40981
    bunch = data = sklearn.datasets.fetch_openml(data_id=40981, as_frame=True)
    y = bunch['target'].to_numpy()
    X = bunch['data'].to_numpy(np.float)

    X_train, X_test, y_train, y_test = \
         sklearn.model_selection.train_test_split(X, y, random_state=1)

    # Auto-sklearn can automatically recognize categorical/numerical data from a pandas
    # DataFrame. This example highlights how the user can provide the feature types,
    # when using numpy arrays, as there is no per-column dtype in this case.
    # feat_type is a list that tags each column from a DataFrame/ numpy array / list
    # with the case-insensitive string categorical or numerical, accordingly.
    feat_type = ['Categorical' if x.name == 'category' else 'Numerical' for x in bunch['data'].dtypes]

    ############################################################################
    # Build and fit a classifier
    # ==========================

    cls = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=30,
        # Bellow two flags are provided to speed up calculations
        # Not recommended for a real implementation
        initial_configurations_via_metalearning=0,
        smac_scenario_args={'runcount_limit': 1},
    )
    cls.fit(X_train, y_train, X_test, y_test, feat_type=feat_type)

    ###########################################################################
    # Get the Score of the final ensemble
    # ===================================

    predictions = cls.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))
