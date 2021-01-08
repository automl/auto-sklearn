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

* `Classification <examples/20_basic/example_classification.html>`_
* `Multi-label Classification <examples/20_basic/example_multilabel_classification.html>`_
* `Regression <examples/20_basic/example_regression.html>`_
* `Continuous and categorical data <examples/40_advanced/example_feature_types.html>`_
* `Iterating over the models <examples/40_advanced/example_get_pipeline_components.html>`_
* `Using custom metrics <examples/40_advanced/example_metrics.html>`_
* `Pandas Train and Test inputs <examples/40_advanced/example_pandas_train_test.html>`_
* `Resampling strategies <examples/40_advanced/example_resampling.html>`_
* `Parallel usage (manual) <examples/60_search/example_parallel_manual_spawning.html>`_
* `Parallel usage (n_jobs) <examples/60_search/example_parallel_n_jobs.html>`_
* `Random search <examples/60_search/example_random_search.html>`_
* `Sequential usage <examples/60_search/example_sequential.html>`_
* `Successive Halving <examples/60_search/example_successive_halving.html>`_
* `Extending with a new classifier <examples/80_extending/example_extending_classification.html>`_
* `Extending with a new regressor <examples/80_extending/example_extending_regression.html>`_
* `Extending with a new preprocessor <examples/80_extending/example_extending_preprocessor.html>`_
* `Restrict hyperparameters for a component <examples/80_extending/example_restrict_number_of_hyperparameters.html>`_


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

Examples for using holdout and cross-validation can be found in `auto-sklearn/examples/ <examples/>`_

Supported Inputs
================
*auto-sklearn* can accept targets for the following tasks (more details on `Sklearn algorithms <https://scikit-learn.org/stable/modules/multiclass.html>`_):
* Binary Classification
* Multiclass Classification
* Multilabel Classification
* Regression
* Multioutput Regression

You can provide feature and target training pairs (X_train/y_train) to *auto-sklearn* to fit an ensemble of pipelines as described in the next section. This X_train/y_train dataset must belong to one of the supported formats: np.ndarray, pd.DataFrame, scipy.sparse.csr_matrix and python lists.
 Optionally, you can measure the ability of this fitted model to generalize to unseen data by providing an optional testing pair (X_test/Y_test). For further details, please refer to the example `Train and Test inputs <examples/example_pandas_train_test.html>`_. Supported formats for these training and testing pairs are: np.ndarray, pd.DataFrame, scipy.sparse.csr_matrix and python lists.

If your data contains categorical values (in the features or targets), autosklearn will automatically encode your data using a `sklearn.preprocessing.LabelEncoder <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html>`_ for unidimensional data and a `sklearn.preprocessing.OrdinalEncoder <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html>`_ for multidimensional data.

Regarding the features, there are two methods to guide *auto-sklearn* to properly encode categorical columns:
* Providing a X_train/X_test numpy array with the optional flag feat_type. For further details, you can check the example `Feature Types <examples/example_feature_types.html>`_.
* You can provide a pandas DataFrame, with properly formatted columns. If a column has numerical dtype, *auto-sklearn* will not encode it and it will be passed directly to scikit-learn. If the column has a categorical/boolean class, it will be encoded. If the column is of any other type (Object or Timeseries), an error will be raised. For further details on how to properly encode your data, you can check the example `Working with categorical data <https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html>`_). If you are working with time series, it is recommended that you follow this approach `Working with time data <https://stats.stackexchange.com/questions/311494/>`_.

Regarding the targets (y_train/y_test), if the task involves a classification problem, such features will be automatically encoded. It is recommended to provide both y_train and y_test during fit, so that a common encoding is created between these splits (if only y_train is provided during fit, the categorical encoder will not be able to handle new classes that are exclusive to y_test). If the task is regression, no encoding happens on the targets.

Ensemble Building Process
=========================

*auto-sklearn* uses ensemble selection by `Caruana et al. (2004) <https://dl.acm.org/doi/pdf/10.1145/1015330.1015432>`_
to build an ensemble based on the models’ prediction for the validation set. The following hyperparameters control how the ensemble is constructed:

* ``ensemble_size`` determines the maximal size of the ensemble. If it is set to zero, no ensemble will be constructed.
* ``ensemble_nbest`` allows the user to directly specify the number of models considered for the ensemble.  This hyperparameter can be an integer *n*, such that only the best *n* models are used in the final ensemble. If a float between 0.0 and 1.0 is provided, ``ensemble_nbest`` would be interpreted as a fraction suggesting the percentage of models to use in the ensemble building process (namely, if ensemble_nbest is a float, library pruning is implemented as described in `Caruana et al. (2006) <https://dl.acm.org/doi/10.1109/ICDM.2006.76>`_).
* ``max_models_on_disc`` defines the maximum number of models that are kept on the disc, as a mechanism to control the amount of disc space consumed by *auto-sklearn*. Throughout the automl process, different individual models are optimized, and their predictions (and other metadata) is stored on disc. The user can set the upper bound on how many models are acceptable to keep on disc, yet this variable takes priority in the definition of the number of models used by the ensemble builder (that is, the minimum of ``ensemble_size``, ``ensemble_nbest`` and ``max_models_on_disc`` determines the maximal amount of models used in the ensemble). If set to None, this feature is disabled.

Inspecting the results
======================

*auto-sklearn* allows users to inspect the training results and statistics. The following example shows how different
statistics can be printed for the inspection.

>>> import autosklearn.classification
>>> automl = autosklearn.classification.AutoSklearnClassifier()
>>> automl.fit(X_train, y_train)
>>> automl.cv_results_
>>> automl.sprint_statistics()
>>> automl.show_models()

``cv_results_`` returns a dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame.
``sprint_statistics()`` is a method that prints the name of the  dataset, the metric used, and the best validation score
obtained by running *auto-sklearn*. It additionally prints the number of both successful and unsuccessful
algorithm runs.

The results obtained from the final ensemble can be printed by calling ``show_models()``. *auto-sklearn* ensemble is composed of scikit-learn models that can be inspected as exemplified by
`model inspection example <examples/example_get_pipeline_components.html>`_
.

Parallel computation
====================

*auto-sklearn* supports parallel execution by data sharing on a shared file
system. In this mode, the SMAC algorithm shares the training data for it's
model by writing it to disk after every iteration. At the beginning of each
iteration, SMAC loads all newly found data points. We provide an example
implementing
`scikit-learn's n_jobs functionality <examples/example_parallel_n_jobs.html>`_
and an example on how
to
`manually start multiple instances of auto-sklearn <examples/example_parallel_manual_spawning.html>`_
.

In it's default mode, *auto-sklearn* already uses two cores. The first one is
used for model building, the second for building an ensemble every time a new
machine learning model has finished training. The
`sequential example <examples/example_sequential.html>`_
shows how to run these tasks sequentially to use only a single core at a time.

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
