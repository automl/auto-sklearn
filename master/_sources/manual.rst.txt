:orphan:

.. _manual:

======
Manual
======

This manual shows how to use several aspects of auto-sklearn. It either
references the examples where possible or explains certain configurations.

Examples
========

*auto-sklearn* comes with the following examples which demonstrate several
aspects of its usage:

* `Holdout <examples/example_holdout.html>`_
* `Cross-validation <examples/example_crossvalidation.html>`_
* `Parallel usage <examples/example_parallel.html>`_
* `Sequential usage <examples/example_sequential.html>`_
* `Regression <examples/example_regression.html>`_
* `Continuous and categorical data <examples/example_feature_types.html>`_
* `Using custom metrics <examples/example_metrics.html>`_
* `Random search <examples/example_random_search.html>`_
* `EIPS <examples/example_eips.html>`_


Time and memory limits
======================

A crucial feature of *auto-sklearn* is limiting the resources (memory and
time) which the scikit-learn algorithms are allowed to use. Especially for
large datasets, on which algorithms can take several hours and make the
machine swap, it is important to stop the evaluations after some time in order
to make progress in a reasonable amount of time. Setting the resource limits
is therefore a tradeoff between optimization time and the number of models
that can be tested.

While *auto-sklearn* alleviates manual hyperparameter tuning, the user still
has to set memory and time limits. For most datasets a memory limit of 3GB or
6GB as found on most modern computers is sufficient. For the time limits it
is harder to give clear guidelines. If possible, a good default is a total
time limit of one day, and a time limit of 30 minutes for a single run.

Further guidelines can be found in
`auto-sklearn/issues/142 <https://github.com/automl/auto-sklearn/issues/142>`_.

Restricting the searchspace
===========================

Instead of using all available estimators, it is possible to restrict
*auto-sklearn*'s searchspace. The following shows an example of how to exclude
all preprocessing methods and restrict the configuration space to only
random forests.

>>> import autosklearn.classification
>>> automl = autosklearn.classification.AutoSklearnClassifier(
>>>     include_estimators=["random_forest", ], exclude_estimators=None,
>>>     include_preprocessors=["no_preprocessing", ], exclude_preprocessors=None)
>>> automl.fit(X_train, y_train)
>>> predictions = automl.predict(X_test)

**Note:** The strings used to identify estimators and preprocessors are the filenames without *.py*.

For a full list please have a look at the source code (in `autosklearn/pipeline/components/`):

  * `Classifiers <https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/classification>`_
  * `Regressors <https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/regression>`_
  * `Preprocessors <https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/feature_preprocessing>`_

Turning off preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~

Preprocessing in *auto-sklearn* is divided into data preprocessing and
feature preprocessing. Data preprocessing includes One-Hot encoding of
categorical features, imputation of missing values and the normalization of
features or samples. These steps currently cannot be turned off. Feature
preprocessing is a single transformer which implements for example feature
selection or transformation of features into a different space (i.e. PCA).
This can be turned off by setting
``include_preprocessors=["no_preprocessing"]`` as shown in the example above.

Resampling strategies
=====================

Examples for using holdout and cross-validation can be found in `auto-sklearn/examples/ <https://github.com/automl/auto-sklearn/tree/master/example>`_

Inspecting the results
======================

*auto-sklearn* allows users to inspect the training results and statistics. The following example shows how different 
statistics can be printed for the inspection.

>>> import autoskleran.classification
>>> automl = autosklearn.classification.AutoSklearnClassifier()
>>> automl.fit(X_train, y_train)
>>> automl.cv_results_
>>> automl.sprint_statistics()
>>> automl.show_models()

``cv_results_`` returns a dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame.
``sprint_statistics()`` is a method that prints the name of the  dataset, the metric used, and the best validation score
obtained by running *auto-sklearn*. It additionally prints the number of both successful and unsuccessful
algorithm runs.
The results obtained from the final ensemble can be printed by calling ``show_models()``.

Parallel computation
====================

*auto-sklearn* supports parallel execution by data sharing on a shared file
system. In this mode, the SMAC algorithm shares the training data for it's
model by writing it to disk after every iteration. At the beginning of each
iteration, SMAC loads all newly found data points. An example can be found in
the example directory.

In it's default mode, *auto-sklearn* already uses two cores. The first one is
used for model building, the second for building an ensemble every time a new
machine learning model has finished training. The file `example_sequential
.py` in the example directory describes how to run these tasks sequentially
to use only a single core at a time.

Furthermore, depending on the installation of scikit-learn and numpy,
the model building procedure may use up to all cores. Such behaviour is
unintended by *auto-sklearn* and is most likely due to numpy being installed
from `pypi` as a binary wheel (`see here <http://scikit-learn-general.narkive
.com/44ywvAHA/binary-wheel-packages-for-linux-are-coming>`_). Executing
``export OPENBLAS_NUM_THREADS=1`` should disable such behaviours and make numpy
only use a single core at a time.

Model persistence
=================

*auto-sklearn* is mostly a wrapper around scikit-learn. Therefore, it is
possible to follow the `persistence example
<http://scikit-learn.org/stable/modules/model_persistence.html#persistence-example>`_
from scikit-learn.

Vanilla auto-sklearn
====================

In order to obtain *vanilla auto-sklearn* as used in `Efficient and Robust Automated Machine Learning
<https://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine -learning>`_
set ``ensemble_size=1`` and ``initial_configurations_via_metalearning=0``:

>>> import autosklearn.classification
>>> automl = autosklearn.classification.AutoSklearnClassifier(
>>>     ensemble_size=1, initial_configurations_via_metalearning=0)

An ensemble of size one will result in always choosing the current best model
according to its performance on the validation set. Setting the initial
configurations found by meta-learning to zero makes *auto-sklearn* use the
regular SMAC algorithm for suggesting new hyperparameter configurations.
