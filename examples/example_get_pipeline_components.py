# -*- encoding: utf-8 -*-
"""
=================================
Query the Classification Pipeline
=================================

The following example shows how to query from a pipeline
built by auto-sklearn
"""
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification

def main():

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    disable_evaluator_output=False,
    resampling_strategy='holdout',
    # We want autosklearn to use pca as preprocessor
    include_preprocessors=['pca'],
)
automl.fit(X_train, y_train, dataset_name='breast_cancer')

predictions = automl.predict(X_test)
# Print statistics about the auto-sklearn run such as number of
# iterations, number of models failed with a time out.
print(automl.sprint_statistics())
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))

# Iterate over the components of the model and print
# The explained variance ratio per stage
for i, (weight, pipeline) in enumerate(automl.get_models_with_weights()):
    for stage_name, component in pipeline.named_steps.items():
        if 'preprocessor' in stage_name:
            print("The {}th pipeline has a explained variance of {}".format(
                i,
                # The component is usually a instance of AutoSklearnChoice.
                # Access the sklearn object via the choice attribute
                # We want the explained variance attributed of
                # each principal component
                component.choice.preprocessor.explained_variance_ratio_
                )
            )

if __name__ == '__main__':
    main()
