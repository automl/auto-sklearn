:orphan:

.. _components:

Available Components
********************

Classification
==============

A list of all classification algorithms considered in the ParamSklearn search space.

.. autoclass:: ParamSklearn.components.classification.adaboost.AdaboostClassifier
    :members:

.. autoclass:: ParamSklearn.components.classification.bernoulli_nb.BernoulliNB
    :members:

.. autoclass:: ParamSklearn.components.classification.extra_trees.ExtraTreesClassifier
    :members:

.. autoclass:: ParamSklearn.components.classification.gaussian_nb.GaussianNB
    :members:

.. autoclass:: ParamSklearn.components.classification.gradient_boosting.GradientBoostingClassifier
    :members:

.. autoclass:: ParamSklearn.components.classification.k_nearest_neighbors.KNearestNeighborsClassifier
    :members:

.. autoclass:: ParamSklearn.components.classification.liblinear.LibLinear_SVC
    :members:
    
.. autoclass:: ParamSklearn.components.classification.libsvm_svc.LibSVM_SVC
    :members:

.. autoclass:: ParamSklearn.components.classification.multinomial_nb.MultinomialNB
    :members:
    
.. autoclass:: ParamSklearn.components.classification.random_forest.RandomForest
    :members:

.. autoclass:: ParamSklearn.components.classification.sgd.SGD
    :members:

Regression
==========

A list of all regression algorithms considered in the ParamSklearn search space.

.. autoclass:: ParamSklearn.components.regression.gaussian_process.GaussianProcess
    :members:

.. autoclass:: ParamSklearn.components.regression.gradient_boosting.GradientBoosting
    :members:

.. autoclass:: ParamSklearn.components.regression.random_forest.RandomForest
    :members:

.. autoclass:: ParamSklearn.components.regression.ridge_regression.RidgeRegression
    :members:


Preprocessing
=============

.. autoclass:: ParamSklearn.components.preprocessing.densifier.Densifier
    :members:

.. autoclass:: ParamSklearn.components.preprocessing.imputation.Imputation
    :members:

.. autoclass:: ParamSklearn.components.preprocessing.kitchen_sinks.RandomKitchenSinks
    :members:

.. autoclass:: ParamSklearn.components.preprocessing.no_preprocessing.NoPreprocessing
    :members:

.. autoclass:: ParamSklearn.components.preprocessing.pca.PCA
    :members:

.. autoclass:: ParamSklearn.components.preprocessing.random_trees_embedding.RandomTreesEmbedding
    :members:

.. autoclass:: ParamSklearn.components.preprocessing.rescaling.Rescaling
    :members:

.. autoclass:: ParamSklearn.components.preprocessing.select_percentile_classification.SelectPercentileClassification
    :members:

.. autoclass:: ParamSklearn.components.preprocessing.select_percentile_regression.SelectPercentileRegression
    :members:

.. autoclass:: ParamSklearn.components.preprocessing.sparse_filtering.SparseFiltering
    :members:

.. autoclass:: ParamSklearn.components.preprocessing.truncatedSVD.TruncatedSVD
