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
from sklearn.datasets import fetch_20newsgroups
import sklearn.metrics

import autosklearn.classification

############################################################################
# Data Loading
# ============

newsgroups_train = fetch_20newsgroups(subset="train", random_state=42, shuffle=True)
newsgroups_test = fetch_20newsgroups(subset="test")

# load train data
df_train = pd.DataFrame({"X": [], "y": []})

for idx, (text, target) in enumerate(
    zip(newsgroups_train.data, newsgroups_train.target)
):
    df_train = pd.concat(
        [
            df_train,
            pd.DataFrame(
                {"X": text, "y": newsgroups_train.target_names[target]}, index=[idx]
            ),
        ]
    )

# explicitly label text column as string
X_train = df_train.astype({"X": "string", "y": "category"})
y_train = X_train.pop("y")

# load test data
df_test = pd.DataFrame({"X": [], "y": []})

for idx, (text, target) in enumerate(zip(newsgroups_test.data, newsgroups_test.target)):
    df_test = pd.concat(
        [
            df_train,
            pd.DataFrame(
                {"X": text, "y": newsgroups_train.target_names[int(target)]},
                index=[idx],
            ),
        ]
    )

# explicitly label text column as string
X_test = df_test.astype({"X": "string", "y": "category"})
y_test = X_test.pop("y")


############################################################################
# Build and fit a classifier
# ==========================

automl = autosklearn.classification.AutoSklearnClassifier(
    # set the time high enough text preprocessing can create many new features
    time_left_for_this_task=300,
    per_run_time_limit=30,
    tmp_folder="/tmp/autosklearn_text_example_tmp",
)
automl.fit(X_train, y_train, dataset_name="20_Newsgroups")

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
