# -*- encoding: utf-8 -*-
"""
=================================
Query the Classification Pipeline
=================================

The following example shows how to query from a pipeline
built by auto-sklearn. Auto-sklearn is a wrapper on top of
the sklearn models. This example illustrates how to interact
with the sklearn components directly, in this case a PCA preprocessor.
"""
import sklearn.model_selection
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
    time_left_for_this_task=120,
    per_run_time_limit=30,
    disable_evaluator_output=False,
    resampling_strategy='holdout',
    # We want autosklearn to use pca as preprocessor
    include_preprocessors=['pca'],
)
automl.fit(X_train, y_train, dataset_name='breast_cancer')

############################################################################
# Report the model found by Auto-Sklearn
# ======================================

predictions = automl.predict(X_test)
# Print statistics about the auto-sklearn run such as number of
# iterations, number of models failed with a time out.
print(automl.sprint_statistics())
print("Accuracy score:{}".format(
    sklearn.metrics.accuracy_score(y_test, predictions))
)

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
