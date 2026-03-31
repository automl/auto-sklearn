# Welcome to auto-sklearn

**auto-sklearn** is an automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator.

## Key Features

- **Automated Model Selection**: Automatically selects the best machine learning model for your data
- **Hyperparameter Optimization**: Optimizes hyperparameters using Bayesian optimization
- **Ensemble Building**: Builds ensembles of models to improve performance
- **Scikit-learn Compatible**: Works seamlessly with scikit-learn pipelines and estimators
- **Parallel Execution**: Supports parallel execution on multiple cores and clusters

## Quick Example

```python
import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

# Load data
X, y = sklearn.datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=1
)

# Create and fit the classifier
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
)
automl.fit(X_train, y_train, dataset_name="digits")

# Evaluate
y_hat = automl.predict(X_test)
print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, y_hat))
```

## Getting Started

- [Installation](getting_started/installation.md)
- [Quick Start](getting_started/quick_start.md)
- [Basic Concepts](getting_started/basic_concepts.md)

## Support

If you encounter any issues or have questions, please:

- Check the [documentation](https://automl.github.io/auto-sklearn)
- Search the [issue tracker](https://github.com/automl/auto-sklearn/issues)
- Open a [new issue](https://github.com/automl/auto-sklearn/issues/new)
