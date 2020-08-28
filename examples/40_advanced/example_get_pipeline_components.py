# -*- encoding: utf-8 -*-
"""
======================
Obtain run information
======================

The following example shows how to obtain information from a finished
Auto-sklearn run. In particular, it shows:
* how to query which models were evaluated by Auto-sklearn
* how to query the models in the final ensemble
* how to get general statistics on the what Auto-sklearn evaluated

Auto-sklearn is a wrapper on top of
the sklearn models. This example illustrates how to interact
with the sklearn components directly, in this case a PCA preprocessor.
"""
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification

############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)

############################################################################
# Build and fit the classifier
# ============================

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=30,
    per_run_time_limit=10,
    disable_evaluator_output=False,
    # To simplify querying the models in the final ensemble, we
    # restrict auto-sklearn to use only pca as a preprocessor
    include_preprocessors=['pca'],
)
automl.fit(X_train, y_train, dataset_name='breast_cancer')

############################################################################
# Predict using the model
# =======================

predictions = automl.predict(X_test)
print("Accuracy score:{}".format(
    sklearn.metrics.accuracy_score(y_test, predictions))
)

############################################################################
# Report the model found by Auto-Sklearn
# ======================================

# Auto-sklearn uses
# `<https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf> Ensemble Selection`_
# to construct ensembles in a post-hoc fashion. The ensemble is a linear
# weighting of all models constructed during the hyperparameter optimization.
# This prints the final ensemble. It is a list of tuples, each tuple being
# the model weight in the ensemble and the model itself.

print(automl.show_models())

###########################################################################
# Report statistics about the search
# ==================================

# Print statistics about the auto-sklearn run such as number of
# iterations, number of models failed with a time out etc.
print(automl.sprint_statistics())

############################################################################
# Detailed statistics about the search - part 1
# =============================================

# Auto-sklearn also keeps detailed statistics of the hyperparameter
# optimization procedurce, which are stored in a so-called
# `<https://automl.github.io/SMAC3/master/apidoc/smac.runhistory.runhistory.html#smac.runhistory# .runhistory.RunHistory> run history`_.

print(automl._automl[0].runhistory_)

# Runs are stored inside an ``OrderedDict`` called ``data``:

print(len(automl._automl[0].runhistory_.data))

# Let's iterative over all entries

for run_key in automl._automl[0].runhistory_.data:
    print('#########')
    print(run_key)
    print(automl._automl[0].runhistory_.data[run_key])

# and have a detailed look at one entry:

run_key = list(automl._automl[0].runhistory_.data.keys())[0]
run_value = automl._automl[0].runhistory_.data[run_key]

# The ``run_key`` contains all information describing a run:

print(run_key.config_id)
print(run_key.instance_id)
print(run_key.seed)
print(run_key.budget)

# and the configuration can be looked up in the run history as well:

print(automl._automl[0].runhistory_.ids_config[run_key.config_id])

# The ``run_value`` contains all output from running the configuration:

print(run_value.cost)
print(run_value.time)
print(run_value.status)
print(run_value.additional_info)
print(run_value.starttime)
print(run_value.endtime)

############################################################################
# Detailed statistics about the search - part 2
# =============================================

# To maintain compatibility with scikit-learn, Auto-sklearn gives the
# same data as
# `<https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html> cv_results_`_.

print(automl.cv_results_)

############################################################################
# Inspect the components of the best model
# ========================================

# Iterate over the components of the model and print
# The explained variance ratio per stage
for i, (weight, pipeline) in enumerate(automl.get_models_with_weights()):
    for stage_name, component in pipeline.named_steps.items():
        if 'preprocessor' in stage_name:
            print(
                "The {}th pipeline has a explained variance of {}".format(
                    i,
                    # The component is an instance of AutoSklearnChoice.
                    # Access the sklearn object via the choice attribute
                    # We want the explained variance attributed of
                    # each principal component
                    component.choice.preprocessor.explained_variance_ratio_
                )
            )
