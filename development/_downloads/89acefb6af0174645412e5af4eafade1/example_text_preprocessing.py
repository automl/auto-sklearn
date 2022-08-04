# -*- encoding: utf-8 -*-
"""
==================
Text preprocessing
==================

The following example shows how to fit a simple NLP problem with
*auto-sklearn*.

For an introduction to text preprocessing you can follow these links:
    1. https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
    2. https://machinelearningmastery.com/clean-text-machine-learning-python/
"""
from pprint import pprint

import pandas as pd
import sklearn.metrics
from sklearn.datasets import fetch_20newsgroups

import autosklearn.classification

############################################################################
# Data Loading
# ============
cats = ["comp.sys.ibm.pc.hardware", "rec.sport.baseball"]
X_train, y_train = fetch_20newsgroups(
    subset="train",  # select train set
    shuffle=True,  # shuffle the data set for unbiased validation results
    random_state=42,  # set a random seed for reproducibility
    categories=cats,  # select only 2 out of 20 labels
    return_X_y=True,  # 20NG dataset consists of 2 columns X: the text data, y: the label
)  # load this two columns separately as numpy array

X_test, y_test = fetch_20newsgroups(
    subset="test",  # select test set for unbiased evaluation
    categories=cats,  # select only 2 out of 20 labels
    return_X_y=True,  # 20NG dataset consists of 2 columns X: the text data, y: the label
)  # load this two columns separately as numpy array

############################################################################
# Creating a pandas dataframe
# ===========================
# Both categorical and text features are often strings. Python Pandas stores python stings
# in the generic `object` type. Please ensure that the correct
# `dtype <https://pandas.pydata.org/docs/user_guide/basics.html#dtypes>`_ is applied to the correct
# column.

# create a pandas dataframe for training labeling the "Text" column as sting
X_train = pd.DataFrame({"Text": pd.Series(X_train, dtype="string")})

# create a pandas dataframe for testing labeling the "Text" column as sting
X_test = pd.DataFrame({"Text": pd.Series(X_test, dtype="string")})

############################################################################
# Build and fit a classifier
# ==========================

# create an autosklearn Classifier or Regressor depending on your task at hand.
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    per_run_time_limit=30,
)

automl.fit(X_train, y_train, dataset_name="20_Newsgroups")  # fit the automl model

############################################################################
# View the models found by auto-sklearn
# =====================================

print(automl.leaderboard())

############################################################################
# Print the final ensemble constructed by auto-sklearn
# ====================================================

pprint(automl.show_models(), indent=4)

###########################################################################
# Get the Score of the final ensemble
# ===================================

predictions = automl.predict(X_test)
print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
