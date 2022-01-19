# -*- encoding: utf-8 -*-
"""
==================
Text Preprocessing
==================
This example shows, how to use text features in *auto-sklearn*. *auto-sklearn* can automatically
encode text features if they are provided as string type in a pandas dataframe.
`https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html`_

For processing text features you need a pandas dataframe and set the desired
text columns to string and the categorical columns to category.

*auto-sklearn* text embedding creates a bag of words count.
`https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html`_
"""
import sklearn.metrics
import sklearn.datasets
import autosklearn.classification

############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.fetch_openml(data_id=40945, return_X_y=True)

# by default, the columns which should be strings are not formatted as such
print(f"{X.info()}\n")

# manually convert these to string columns
X = X.astype({'name': 'string', 'ticket': 'string', 'cabin': 'string', 'boat': 'string',
              'home.dest': 'string'})

# now *auto-sklearn* handles the string columns with its text feature preprocessing pipeline

X_train, X_test, y_train, y_test = \
     sklearn.model_selection.train_test_split(X, y, random_state=1)

cls = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=30,
    # Bellow two flags are provided to speed up calculations
    # Not recommended for a real implementation
    initial_configurations_via_metalearning=0,
    smac_scenario_args={'runcount_limit': 1},
)

cls.fit(X_train, y_train, X_test, y_test)

predictions = cls.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))


X, y = sklearn.datasets.fetch_openml(data_id=40945, return_X_y=True, as_frame=True)
X = X.select_dtypes(exclude=['object'])

X_train, X_test, y_train, y_test = \
     sklearn.model_selection.train_test_split(X, y, random_state=1)

cls = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=30,
    # Bellow two flags are provided to speed up calculations
    # Not recommended for a real implementation
    initial_configurations_via_metalearning=0,
    smac_scenario_args={'runcount_limit': 1},
)

cls.fit(X_train, y_train, X_test, y_test)

predictions = cls.predict(X_test)
print("Accuracy score without text preprocessing", sklearn.metrics.accuracy_score(y_test, predictions))
