:orphan:

.. _components:

Available Components
********************

Classification
==============

A list of all classification algorithms considered in the AutoSklearn search space.

.. autoclass:: AutoSklearn.components.classification.extra_trees.ExtraTreesClassifier
    :members:

.. autoclass:: AutoSklearn.components.classification.gradient_boosting.GradientBoostingClassifier
    :members:

.. autoclass:: AutoSklearn.components.classification.k_nearest_neighbors.KNearestNeighborsClassifier
    :members:

.. autoclass:: AutoSklearn.components.classification.liblinear.LibLinear_SVC
    :members:
    
.. autoclass:: AutoSklearn.components.classification.libsvm_svc.LibSVM_SVC
    :members:
    
.. autoclass:: AutoSklearn.components.classification.random_forest.RandomForest
    :members:

.. autoclass:: AutoSklearn.components.classification.sgd.SGD
    :members:

Regression
==========

Currently there is no AutoSklearnRegressor.

Preprocessing
=============

.. autoclass:: AutoSklearn.components.preprocessing.pca.PCA
