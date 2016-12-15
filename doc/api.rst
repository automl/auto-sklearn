:orphan:

.. _api:

APIs
****

Main modules
============

Classification
~~~~~~~~~~~~~~

.. autoclass:: autosklearn.classification.AutoSklearnClassifier
    :members:
    :inherited-members: show_models, fit_ensemble, refit

Regression
~~~~~~~~~~

.. autoclass:: autosklearn.regression.AutoSklearnRegressor
    :members:
    :inherited-members: show_models, fit_ensemble, refit

Extension Interfaces
====================

.. autoclass:: autosklearn.pipeline.components.base.AutoSklearnClassificationAlgorithm
    :members:

.. autoclass:: autosklearn.pipeline.components.base.AutoSklearnRegressionAlgorithm
    :members:

.. autoclass:: autosklearn.pipeline.components.base.AutoSklearnPreprocessingAlgorithm
    :members:
