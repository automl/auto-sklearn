:orphan:

.. _api:

APIs
****

============
Main modules
============

~~~~~~~~~~~~~~
Classification
~~~~~~~~~~~~~~

.. autoclass:: autosklearn.classification.AutoSklearnClassifier
    :members:
    :inherited-members: show_models, fit_ensemble, refit, sprint_statistics

~~~~~~~~~~
Regression
~~~~~~~~~~

.. autoclass:: autosklearn.regression.AutoSklearnRegressor
    :members:
    :inherited-members: show_models, fit_ensemble, refit, sprint_statistics

=======
Metrics
=======

.. autofunction:: autosklearn.metrics.make_scorer

~~~~~~~~~~~~~~~~
Built-in Metrics
~~~~~~~~~~~~~~~~

Classification metrics
~~~~~~~~~~~~~~~~~~~~~~
Note: The default ``autosklearn.metrics.f1``, ``autosklearn.metrics.precision`` and ``autosklearn.metrics.recall``
built-in metrics are applicable only for binary classification. In order to apply them on multilabel and multiclass
classification, please use the corresponding metrics with an appropriate averaging mechanism, such as ``autosklearn.metrics.f1_macro``.
For more information about how these metrics are used, please read
`this scikit-learn documentation <http://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures>`_.

.. autoclass:: autosklearn.metrics.accuracy

.. autoclass:: autosklearn.metrics.balanced_accuracy

.. autoclass:: autosklearn.metrics.f1

.. autoclass:: autosklearn.metrics.f1_macro

.. autoclass:: autosklearn.metrics.f1_micro

.. autoclass:: autosklearn.metrics.f1_samples

.. autoclass:: autosklearn.metrics.f1_weighted

.. autoclass:: autosklearn.metrics.roc_auc

.. autoclass:: autosklearn.metrics.precision

.. autoclass:: autosklearn.metrics.precision_macro

.. autoclass:: autosklearn.metrics.precision_micro

.. autoclass:: autosklearn.metrics.precision_samples

.. autoclass:: autosklearn.metrics.precision_weighted

.. autoclass:: autosklearn.metrics.average_precision

.. autoclass:: autosklearn.metrics.recall

.. autoclass:: autosklearn.metrics.recall_macro

.. autoclass:: autosklearn.metrics.recall_micro

.. autoclass:: autosklearn.metrics.recall_samples

.. autoclass:: autosklearn.metrics.recall_weighted

.. autoclass:: autosklearn.metrics.log_loss

.. autoclass:: autosklearn.metrics.pac_score

Regression metrics
~~~~~~~~~~~~~~~~~~

.. autoclass:: autosklearn.metrics.r2

.. autoclass:: autosklearn.metrics.mean_squared_error

.. autoclass:: autosklearn.metrics.mean_absolute_error

.. autoclass:: autosklearn.metrics.median_absolute_error

====================
Extension Interfaces
====================

.. autoclass:: autosklearn.pipeline.components.base.AutoSklearnClassificationAlgorithm
    :members:

.. autoclass:: autosklearn.pipeline.components.base.AutoSklearnRegressionAlgorithm
    :members:

.. autoclass:: autosklearn.pipeline.components.base.AutoSklearnPreprocessingAlgorithm
    :members:


