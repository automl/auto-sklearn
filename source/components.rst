:orphan:

.. _components:

Available Components
********************

Classification
==============

A list of all classification algorithms considered in the autosklearn.pipeline search space.

.. autoclass:: autosklearn.pipeline.components.classification.adaboost.AdaboostClassifier
    :members:

.. autoclass:: autosklearn.pipeline.components.classification.bernoulli_nb.BernoulliNB
    :members:

.. autoclass:: autosklearn.pipeline.components.classification.extra_trees.ExtraTreesClassifier
    :members:

.. autoclass:: autosklearn.pipeline.components.classification.gaussian_nb.GaussianNB
    :members:

.. autoclass:: autosklearn.pipeline.components.classification.gradient_boosting.GradientBoostingClassifier
    :members:

.. autoclass:: autosklearn.pipeline.components.classification.k_nearest_neighbors.KNearestNeighborsClassifier
    :members:
    
.. autoclass:: autosklearn.pipeline.components.classification.libsvm_svc.LibSVM_SVC
    :members:

.. autoclass:: autosklearn.pipeline.components.classification.multinomial_nb.MultinomialNB
    :members:
    
.. autoclass:: autosklearn.pipeline.components.classification.random_forest.RandomForest
    :members:

.. autoclass:: autosklearn.pipeline.components.classification.sgd.SGD
    :members:

Regression
==========

A list of all regression algorithms considered in the autosklearn.pipeline search space.

.. autoclass:: autosklearn.pipeline.components.regression.gaussian_process.GaussianProcess
    :members:

.. autoclass:: autosklearn.pipeline.components.regression.gradient_boosting.GradientBoosting
    :members:

.. autoclass:: autosklearn.pipeline.components.regression.random_forest.RandomForest
    :members:

.. autoclass:: autosklearn.pipeline.components.regression.ridge_regression.RidgeRegression
    :members:


Preprocessing
=============

.. autoclass:: autosklearn.pipeline.components.feature_preprocessing.kitchen_sinks.RandomKitchenSinks
    :members:

.. autoclass:: autosklearn.pipeline.components.feature_preprocessing.no_preprocessing.NoPreprocessing
    :members:

.. autoclass:: autosklearn.pipeline.components.feature_preprocessing.pca.PCA
    :members:

.. autoclass:: autosklearn.pipeline.components.feature_preprocessing.random_trees_embedding.RandomTreesEmbedding
    :members:

.. autoclass:: autosklearn.pipeline.components.feature_preprocessing.select_percentile_classification.SelectPercentileClassification
    :members:

.. autoclass:: autosklearn.pipeline.components.feature_preprocessing.select_percentile_regression.SelectPercentileRegression
    :members:

.. autoclass:: autosklearn.pipeline.components.feature_preprocessing.truncatedSVD.TruncatedSVD
    :members:
