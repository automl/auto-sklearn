:orphan:

.. _manual:

======
Manual
======

This manual gives an overview of different aspects of *auto-sklearn*. For each section, we either references examples or
give short explanations (click the title to expand text), e.g.

.. collapse:: <b>Code examples</b>

    We provide examples on using *auto-sklearn* for multiple use cases ranging from
    simple classification to advanced uses such as feature importance, parallel runs
    and customization. They can be found in the :ref:`examples`.

.. collapse:: <b>Material from talks and presentations</b>

    We provide resources for talks, tutorials and presentations on *auto-sklearn* under `auto-sklearn-talks <https://github.com/automl/auto-sklearn-talks>`_

.. _askl2:

Auto-sklearn 2.0
================

Auto-sklearn 2.0 includes latest research on automatically configuring the AutoML system itself
and contains a multitude of improvements which speed up the fitting the AutoML system.
Concretely, Auto-sklearn 2.0 automatically sets the :ref:`bestmodel`, decides whether it can use
the efficient bandit strategy *Successive Halving* and uses meta-feature free *Portfolios* for
efficient meta-learning.

*auto-sklearn 2.0* has the same interface as regular *auto-sklearn* and you can use it via

.. code:: python

    from autosklearn.experimental.askl2 import AutoSklearn2Classifier

A paper describing our advances is available on `arXiv <https://arxiv.org/abs/2007.04074>`_.

.. _limits:

Resource limits
===============

A crucial feature of *auto-sklearn* is limiting the resources (memory and time) which the scikit-learn algorithms are
allowed to use. Especially for large datasets, on which algorithms can take several hours and make the machine swap,
it is important to stop the evaluations after some time in order to make progress in a reasonable amount of time.
Setting the resource limits is therefore a tradeoff between optimization time and the number of models that can be
tested.

.. collapse:: <b>Time and memory limits</b>

    While *auto-sklearn* alleviates manual hyperparameter tuning, the user still
    has to set memory and time limits. For most datasets a memory limit of 3GB or
    6GB as found on most modern computers is sufficient. For the time limits it
    is harder to give clear guidelines. If possible, a good default is a total
    time limit of one day, and a time limit of 30 minutes for a single run.

    Further guidelines can be found in
    `auto-sklearn/issues/142 <https://github.com/automl/auto-sklearn/issues/142>`_.

.. collapse:: <b>CPU cores</b>

    By default, *auto-sklearn* uses **one core**. See also :ref:`parallel` on how to configure this.


.. collapse:: <b>Managing data compression</b>

    .. _manual_managing_data_compression:

    Auto-sklearn will attempt to fit the dataset into 1/10th of the ``memory_limit``.
    This won't happen unless your dataset is quite large or you have small a
    ``memory_limit``. This is done using two methods, reducing **precision** and
    to **subsample**. One reason you may want to control this is if you require high
    precision or you rely on predefined splits for which subsampling does not account
    for.

    To turn off data preprocessing:

    .. code:: python

        AutoSklearnClassifier(
            dataset_compression = False
        )

    You can specify which of the methods are performed using:

    .. code:: python

        AutoSklearnClassifier(
            dataset_compression = { "methods": ["precision", "subsample"] },
        )

    You can change the memory allocation for the dataset to a percentage of ``memory_limit``
    or an absolute amount using:

    .. code:: python

        AutoSklearnClassifier(
            dataset_compression = { "memory_allocation": 0.2 },
        )

    The default arguments are used when ``dataset_compression = True`` are:

    .. code:: python

        {
            "memory_allocation": 0.1,
            "methods": ["precision", "subsample"]
        }

    The full description is given at :class:`AutoSklearnClassifier(dataset_compression=...) <autosklearn.classification.AutoSklearnClassifier>`.

.. _space:

The search space
================

*Auto-sklearn* by default searches a large space to find a well performing configuration. However, it is also possible
to restrict the searchspace:

.. collapse:: <b>Restricting the searchspace</b>

 The following shows an example of how to exclude all preprocessing methods and restrict the configuration space to
 only random forests.

    .. code:: python

        import autosklearn.classification
        automl = autosklearn.classification.AutoSklearnClassifier(
            include = {
                'classifier': ["random_forest"],
                'feature_preprocessor': ["no_preprocessing"]
            },
            exclude=None
        )
        automl.fit(X_train, y_train)
        predictions = automl.predict(X_test)

    **Note:** The strings used to identify estimators and preprocessors are the filenames without *.py*.

    For a full list please have a look at the source code (in `autosklearn/pipeline/components/`):

      * `Classifiers <https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/classification>`_
      * `Regressors <https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/regression>`_
      * `Preprocessors <https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/feature_preprocessing>`_

    We do also provide an example on how to restrict the classifiers to search over
    :ref:`sphx_glr_examples_40_advanced_example_interpretable_models.py`.

.. collapse:: <b>Turn off data preprocessing</b>

    Data preprocessing includes One-Hot encoding of categorical features, imputation
    of missing values and the normalization of features or samples. These ensure that
    the data the gets to the sklearn models is well formed and can be used for
    training models.

    While this is necessary in general, if you'd like to disable this step, please
    refer to this :ref:`example <sphx_glr_examples_80_extending_example_extending_data_preprocessor.py>`.

.. collapse:: <b>Turn off feature preprocessing</b>

    Feature preprocessing is a single transformer which implements for example feature
    selection or transformation of features into a different space (i.e. PCA).

    This can be turned off by setting
    ``include={'feature_preprocessor'=["no_preprocessing"]}`` as shown in the example above.

.. _bestmodel:

Model selection
===============

*Auto-sklearn* implements different strategies to identify the best performing model. For some use cases it might be
necessary to adapt the resampling strategy or define a custom metric:

.. collapse:: <b>Use different resampling strategies</b>

    Examples for using holdout and cross-validation can be found in :ref:`example <sphx_glr_examples_40_advanced_example_resampling.py>`

.. collapse:: <b>Use a custom metric</b>

    Examples for using a custom metric can be found in :ref:`example <sphx_glr_examples_40_advanced_example_metrics.py>`

.. _ensembles:

Ensembling
==========

To get the best performance out of the evaluated models, *auto-sklearn* uses ensemble selection by `Caruana et al. (2004) <https://dl.acm.org/doi/pdf/10.1145/1015330.1015432>`_
to build an ensemble based on the models’ prediction for the validation set.

.. collapse:: <b>Configure the ensemble building process</b>

    The following hyperparameters control how the ensemble is constructed:

    * ``ensemble_size`` determines the maximal size of the ensemble. If it is set to zero, no ensemble will be constructed.
    * ``ensemble_nbest`` allows the user to directly specify the number of models considered for the ensemble.  This hyperparameter can be an integer *n*, such that only the best *n* models are used in the final ensemble. If a float between 0.0 and 1.0 is provided, ``ensemble_nbest`` would be interpreted as a fraction suggesting the percentage of models to use in the ensemble building process (namely, if ensemble_nbest is a float, library pruning is implemented as described in `Caruana et al. (2006) <https://dl.acm.org/doi/10.1109/ICDM.2006.76>`_).
    * ``max_models_on_disc`` defines the maximum number of models that are kept on the disc, as a mechanism to control the amount of disc space consumed by *auto-sklearn*. Throughout the automl process, different individual models are optimized, and their predictions (and other metadata) is stored on disc. The user can set the upper bound on how many models are acceptable to keep on disc, yet this variable takes priority in the definition of the number of models used by the ensemble builder (that is, the minimum of ``ensemble_size``, ``ensemble_nbest`` and ``max_models_on_disc`` determines the maximal amount of models used in the ensemble). If set to None, this feature is disabled.

.. collapse:: <b>Inspect the final ensemble</b>

    The results obtained from the final ensemble can be printed by calling ``show_models()``.
    The *auto-sklearn* ensemble is composed of scikit-learn models that can be inspected as exemplified
    in the Example :ref:`sphx_glr_examples_40_advanced_example_get_pipeline_components.py`.

.. collapse:: <b>Fit ensemble post-hoc</b>

    To use a single core only, it is possible to build ensembles post-hoc. An example on how to do this (first searching
    for individual models, and then building an ensemble from them) can be seen in
    :ref:`sphx_glr_examples_60_search_example_sequential.py`.


.. _inspect:

Inspecting the results
======================

*auto-sklearn* allows users to inspect the training results and statistics. Assume we have a fitted estimator:

.. code:: python

        import autosklearn.classification
        automl = autosklearn.classification.AutoSklearnClassifier()
        automl.fit(X_train, y_train)

*auto-sklearn* offers the following ways to inspect the results

.. collapse:: <b>Basic statistics</b>

    ``sprint_statistics()`` is a method that prints the name of the  dataset, the metric used, and the best validation score
    obtained by running *auto-sklearn*. It additionally prints the number of both successful and unsuccessful
    algorithm runs.

.. collapse:: <b>Performance over Time</b>

    ``performance_over_time_``  returns a DataFrame containing the models performance over time data, which can
    be used for plotting directly (Here is an example: :ref:`sphx_glr_examples_40_advanced_example_pandas_train_test.py`).

    .. code:: python

        automl.performance_over_time_.plot(
                x='Timestamp',
                kind='line',
                legend=True,
                title='Auto-sklearn accuracy over time',
                grid=True,
            )
            plt.show()

.. collapse:: <b>Evaluated models</b>

    The results obtained from the final ensemble can be printed by calling ``show_models()``.

.. collapse:: <b>Leaderboard</b>

    ``automl.leaderboard()`` shows the ensemble members, check the :meth:`docs <autosklearn.classification.AutoSklearnClassifier.leaderboard>` for using leaderboard for getting information on *all* runs.

.. collapse:: <b>Other</b>

    ``cv_results_`` returns a dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame.

.. _parallel:

Parallel computation
====================

In it's default mode, *auto-sklearn* uses **one core** and interleaves ensemble building with evaluating new
configurations.

.. collapse:: <b>Parallelization with Dask</b>

    Nevertheless, *auto-sklearn* also supports parallel Bayesian optimization via the use of
    `Dask.distributed  <https://distributed.dask.org/>`_. By providing the arguments ``n_jobs``
    to the estimator construction, one can control the number of cores available to *auto-sklearn*
    (As shown in the Example :ref:`sphx_glr_examples_60_search_example_parallel_n_jobs.py`).
    Distributed processes are also supported by providing a custom client object to *auto-sklearn* like
    in the Example: :ref:`sphx_glr_examples_60_search_example_parallel_manual_spawning_cli.py`. When
    multiple cores are
    available, *auto-sklearn* will create a worker per core, and use the available workers to both search
    for better machine learning models as well as building an ensemble with them until the time resource
    is exhausted.

    **Note:** *auto-sklearn* requires all workers to have access to a shared file system for storing training data and models.

    *auto-sklearn* employs `threadpoolctl <https://github.com/joblib/threadpoolctl/>`_ to control the number of threads employed by scientific libraries like numpy or scikit-learn. This is done exclusively during the building procedure of models, not during inference. In particular, *auto-sklearn* allows each pipeline to use at most 1 thread during training. At predicting and scoring time this limitation is not enforced by *auto-sklearn*. You can control the number of resources
    employed by the pipelines by setting the following variables in your environment, prior to running *auto-sklearn*:

    .. code-block:: shell-session

        $ export OPENBLAS_NUM_THREADS=1
        $ export MKL_NUM_THREADS=1
        $ export OMP_NUM_THREADS=1


    For further information about how scikit-learn handles multiprocessing, please check the `Parallelism, resource management, and configuration <https://scikit-learn.org/stable/computing/parallelism.html>`_ documentation from the library.

.. _othermanual:

Other
=====

.. collapse:: <b>Supported input types</b>

    *auto-sklearn* can accept targets for the following tasks (more details on `Sklearn algorithms <https://scikit-learn.org/stable/modules/multiclass.html>`_):

    * Binary Classification
    * Multiclass Classification
    * Multilabel Classification
    * Regression
    * Multioutput Regression

    You can provide feature and target training pairs (X_train/y_train) to *auto-sklearn* to fit an
    ensemble of pipelines as described in the next section. This X_train/y_train dataset must belong
    to one of the supported formats: np.ndarray, pd.DataFrame, scipy.sparse.csr_matrix and python lists.
    Optionally, you can measure the ability of this fitted model to generalize to unseen data by
    providing an optional testing pair (X_test/Y_test). For further details, please refer to the
    Example :ref:`sphx_glr_examples_40_advanced_example_pandas_train_test.py`.

    Regarding the features, there are multiple things to consider:

    * Providing a X_train/X_test numpy array with the optional flag feat_type. For further details, you
      can check the Example :ref:`sphx_glr_examples_40_advanced_example_feature_types.py`.
    * You can provide a pandas DataFrame with properly formatted columns. If a column has numerical
      dtype, *auto-sklearn* will not encode it and it will be passed directly to scikit-learn. *auto-sklearn*
      supports both categorical or string as column type. Please ensure that you are using the correct
      dtype for your task. By default *auto-sklearn* treats object and string columns as strings and
      encodes the data using `sklearn.feature_extraction.text.CountVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_
    * If your data contains categorical values (in the features or targets), ensure that you explicitly label them as categorical.
      Data labeled as categorical is encoded by using a `sklearn.preprocessing.LabelEncoder <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html>`_
      for unidimensional data and a `sklearn.preprodcessing.OrdinalEncoder <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html>`_ for multidimensional data.
    * For further details on how to properly encode your data, you can check the Pandas Example
      `Working with categorical data <https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html>`_). If you are working with time series, it is recommended that you follow this approach
      `Working with time data <https://stats.stackexchange.com/questions/311494/>`_.
    * If you prefer not using the string option at all you can disable this option. In this case
      objects, strings and categorical columns are encoded as categorical.

    .. code:: python

        import autosklearn.classification
        automl = autosklearn.classification.AutoSklearnClassifier(allow_string_features=False)
        automl.fit(X_train, y_train)

    Regarding the targets (y_train/y_test), if the task involves a classification problem, such features will be
    automatically encoded. It is recommended to provide both y_train and y_test during fit, so that a common encoding
    is created between these splits (if only y_train is provided during fit, the categorical encoder will not be able
    to handle new classes that are exclusive to y_test). If the task is regression, no encoding happens on the
    targets.

.. collapse:: <b>Model persistence</b>

    *auto-sklearn* is mostly a wrapper around scikit-learn. Therefore, it is
    possible to follow the
    `persistence Example <https://scikit-learn.org/stable/modules/model_persistence.html>`_
    from scikit-learn.

.. collapse:: <b>Vanilla auto-sklearn</b>

    In order to obtain *vanilla auto-sklearn* as used in `Efficient and Robust Automated Machine Learning
    <https://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine -learning>`_
    set ``ensemble_size=1``, ``initial_configurations_via_metalearning=0`` and ``allow_string_features=False``:

    .. code:: python

        import autosklearn.classification
        automl = autosklearn.classification.AutoSklearnClassifier(
            ensemble_size=1,
            initial_configurations_via_metalearning=0,
            allow_string_features=False,
        )

    An ensemble of size one will result in always choosing the current best model
    according to its performance on the validation set. Setting the initial
    configurations found by meta-learning to zero makes *auto-sklearn* use the
    regular SMAC algorithm for suggesting new hyperparameter configurations.

.. collapse:: <b>Early stopping and Callbacks</b>

   By using the parameter ``get_trials_callback``, we can get access to the results
   of runs as they occur. See this example :ref:`Early Stopping And Callbacks <sphx_glr_examples_40_advanced_example_early_stopping_and_callbacks.py>` for more!
