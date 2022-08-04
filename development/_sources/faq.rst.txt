:orphan:

.. _faq:

===
FAQ
===

General
=======

.. collapse:: <b>Where can I find examples on how to use auto-sklearn?</b>

    We provide examples on using *auto-sklearn* for multiple use cases ranging from
    simple classification to advanced uses such as feature importance, parallel runs
    and customization. They can be found in the :ref:`examples`.

.. collapse:: <b>What type of tasks can auto-sklearn tackle?</b>

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

.. collapse:: <b>Where can I find slides and notebooks from talks and tutorials?</b>

    We provide resources for talks, tutorials and presentations on *auto-sklearn* under `auto-sklearn-talks <https://github.com/automl/auto-sklearn-talks>`_

.. collapse:: <b>How should I cite auto-sklearn in a scientific publication?</b>

    If you've used auto-sklearn in scientific publications, we would appreciate citations.

    .. code-block::

        @inproceedings{feurer-neurips15a,
            title     = {Efficient and Robust Automated Machine Learning},
            author    = {Feurer, Matthias and Klein, Aaron and Eggensperger, Katharina  Springenberg, Jost and Blum, Manuel and Hutter, Frank},
            booktitle = {Advances in Neural Information Processing Systems 28 (2015)},
            pages     = {2962--2970},
            year      = {2015}
        }

    Or this, if you've used auto-sklearn 2.0 in your work:

    .. code-block::

        @article{feurer-arxiv20a,
            title     = {Auto-Sklearn 2.0: Hands-free AutoML via Meta-Learning},
            author    = {Feurer, Matthias and Eggensperger, Katharina and Falkner, Stefan and Lindauer, Marius and Hutter, Frank},
            booktitle = {arXiv:2007.04074 [cs.LG]},
            year      = {2020}
        }

.. collapse:: <b>I want to contribute. What can I do?</b>

    This sounds great. Please have a look at our `contribution guide <https://github.com/automl/auto-sklearn/blob/master/CONTRIBUTING.md>`_

.. collapse:: <b>I have a question which is not answered here. What should I do?</b>

    Thanks a lot. We regularly update this section with questions from our issue tracker. So please use the
    `issue tracker <https://github.com/automl/auto-sklearn/issues>`_

Resource Management
===================

.. collapse:: <b>How should I set the time and memory limits?</b>

    While *auto-sklearn* alleviates manual hyperparameter tuning, the user still
    has to set memory and time limits. For most datasets a memory limit of 3GB or
    6GB as found on most modern computers is sufficient. For the time limits it
    is harder to give clear guidelines. If possible, a good default is a total
    time limit of one day, and a time limit of 30 minutes for a single run.

    Further guidelines can be found in
    `auto-sklearn/issues/142 <https://github.com/automl/auto-sklearn/issues/142>`_.

.. collapse:: <b>How many CPU cores does auto-sklearn use by default?</b>

    By default, *auto-sklearn* uses **one core**. See also :ref:`parallel` on how to configure this.

.. collapse:: <b>How can I run auto-sklearn in parallel?</b>

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


.. collapse:: <b>Auto-sklearn is extremely memory hungry in a sequential setting</b>

    Auto-sklearn can appear very memory hungry (i.e. requiring a lot of memory for small datasets) due
    to the use of ``fork`` for creating new processes when running in sequential manner (if this
    happens in a parallel setting or if you pass your own dask client this is due to a different
    issue, see the other issues below).

    Let's go into some more detail and discuss how to fix it:
    Auto-sklearn executes each machine learning algorithm in its own process to be able to apply a
    memory limit and a time limit. To start such a process, Python gives three options: ``fork``,
    ``forkserver`` and ``spawn``. The default ``fork`` copies the whole process memory into the
    subprocess. If the main process already uses 1.5GB of main memory and we apply a 3GB memory
    limit to Auto-sklearn, executing a machine learning pipeline is limited to use at most 1.5GB.
    We would have loved to use ``forkserver`` or ``spawn`` as the default option instead, which both
    copy only relevant data into the subprocess and thereby alleaviate the issue of eating up a lot
    of your main memory
    (and also do not suffer from potential deadlocks as ``fork`` does, see
    `here <https://pythonspeed.com/articles/python-multiprocessing/>`_),
    but they have the downside that code must be guarded by ``if __name__ == "__main__"`` or executed
    in a notebook, and we decided that we do not want to require this by default.

    There are now two possible solutions:

    1. Use Auto-sklearn in parallel: if you use Auto-sklean in parallel, it defaults to ``forkserver``
       as the parallelization mechanism itself requires Auto-sklearn the code to be guarded. Please
       find more information on how to do this in the following two examples:

       1. :ref:`sphx_glr_examples_60_search_example_parallel_n_jobs.py`
       2. :ref:`sphx_glr_examples_60_search_example_parallel_manual_spawning_cli.py`

       .. note::

           This requires all code to be guarded by ``if __name__ == "__main__"``.

    2. Pass a `dask client <https://distributed.dask.org/en/latest/client.html>`_. If the user passes
       a dask client, Auto-sklearn can no longer assume that it runs in sequential mode and will use
       a ``forkserver`` to start new processes.

       .. note::

           This requires all code to be guarded by ``if __name__ == "__main__"``.

    We therefore suggest using one of the above settings by default.

.. collapse:: <b>Auto-sklearn is extremely memory hungry in a parallel setting</b>

    When running Auto-sklearn in a parallel setting it starts new processes for evaluating machine
    learning models using the ``forkserver`` mechanism. Code that is in the main script and that is
    not guarded by ``if __name__ == "__main__"`` will be executed for each subprocess. If, for example,
    you are loading your dataset outside of the guarded code, your dataset will be loaded for each
    evaluation of a machine learning algorithm and thus blocking your RAM.

    We therefore suggest moving all code inside functions or the main block.

.. collapse:: <b>Auto-sklearn crashes with a segmentation fault</b>

    Please make sure that you have read and followed the :ref:`installation` section! In case
    everything is set up correctly, this is most likely due to the dependency
    `pyrfr <https://github.com/automl/random_forest_run>`_ not being compiled correctly. If this is the
    case please execute:

    .. code:: python

        import pyrfr.regression as reg
        data = reg.default_data_container(64)

    If this fails, the pyrfr dependency is most likely not compiled correctly. We advice you to do the
    following:

    1. Check if you can use a pre-compiled version of the pyrfr to avoid compiling it yourself. We
       provide pre-compiled versions of the pyrfr on `pypi <https://pypi.org/project/pyrfr/#files>`_.
    2. Check if the dependencies specified under :ref:`installation` are correctly installed,
       especially that you have ``swig`` and a ``C++`` compiler.
    3. If you are not yet using Conda, consider using it; it simplifies installation of the correct
       dependencies.
    4. Install correct build dependencies before installing the pyrfr, you can check the following
       github issues for suggestions: `1025 <https://github.com/automl/auto-sklearn/issues/1025>`_,
       `856 <https://github.com/automl/auto-sklearn/issues/856>`_

Results, Log Files and Output
=============================

.. collapse:: <b>How can I get an overview of the run statistics?</b>

    ``sprint_statistics()`` is a method that prints the name of the  dataset, the metric used, and the best validation score
    obtained by running *auto-sklearn*. It additionally prints the number of both successful and unsuccessful
    algorithm runs.

.. collapse:: <b>What was the performance over time?</b>

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

.. collapse:: <b>Which models were evaluated?</b>

    You can see all models evaluated using :meth:`automl.leaderboard(ensemble_only=False) <autosklearn.classification.AutoSklearnClassifier.leaderboard>`.

.. collapse:: <b>Which models are in the final ensemble?</b>

    Use either :meth:`automl.leaderboard(ensemble_only=True) <autosklearn.classification.AutoSklearnClassifier.leaderboard>` or ``automl.show_models()``

.. collapse:: <b>Is there more data I can look at?</b>

    ``cv_results_`` returns a dict with keys as column headers and values as columns, that can be imported into
    a pandas DataFrame, e.g. ``df = pd.DataFrame(automl.cv_results_)``

.. collapse:: <b>Where does Auto-sklearn output files by default?</b>

    *Auto-sklearn* heavily uses the hard drive to store temporary data, models and log files which can
    be used to inspect the behavior of Auto-sklearn. Each run of Auto-sklearn requires
    its own directory. If not provided by the user, *Auto-sklearn* requests a temporary directory from
    Python, which by default is located under ``/tmp`` and starts with ``autosklearn_tmp_`` followed
    by a random string. By default, this directory is deleted when the *Auto-sklearn* object is
    finished fitting. If you want to keep these files you can pass the argument
    ``delete_tmp_folder_after_terminate=True`` to the *Auto-sklearn* object.

    The :class:`autosklearn.classification.AutoSklearnClassifier` and all other *auto-sklearn*
    estimators accept the argument ``tmp_folder`` which change where such output is written to.

    There's an additional argument ``output_directory`` which can be passed to *Auto-sklearn* and it
    controls where test predictions of the ensemble are stored if the test set is passed to ``fit()``.

.. collapse:: <b>Auto-sklearn's logfiles eat up all my disk space. What can I do?</b>

    *Auto-sklearn* heavily uses the hard drive to store temporary data, models and log files which can
    be used to inspect the behavior of Auto-sklearn. By default, *Auto-sklearn* stores 50
    models and their predictions on the validation data (which is a subset of the training data in
    case of holdout and the full training data in case of cross-validation) on the hard drive.
    Redundant models and their predictions (i.e. when we have more than 50 models) are removed
    everytime the ensemble builder finishes an iteration, which means that the number of models stored
    on disk can temporarily be higher if a model is output while the ensemble builder is running.

    One can therefore change the number of models that will be stored on disk by passing an integer
    for the argument ``max_models_on_disc`` to *Auto-sklearn*, for example reduce the number of models
    stored on disk if you have space issues.

    As the number of models is only an indicator of the disk space used it is also possible to pass
    the memory in MB the models are allowed to use as a ``float`` (also via the ``max_models_on_disc``
    arguments). As above, this is rather a guideline on how much memory is used as redundant models
    are only removed from disk when the ensemble builder finishes an iteration.

    .. note::

        Especially when running in parallel it can happen that multiple models are constructed during
        one run of the ensemble builder and thus *Auto-sklearn* can exceed the given limit.

    .. note::

       These limits do only apply to models and their predictions, but not to other files stored in
       the temporary directory such as the log files.

The Search Space
================

.. collapse:: <b>How can I restrict the searchspace?</b>

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

.. collapse:: <b>How can I turn off data preprocessing?</b>

    Data preprocessing includes One-Hot encoding of categorical features, imputation
    of missing values and the normalization of features or samples. These ensure that
    the data the gets to the sklearn models is well formed and can be used for
    training models.

    While this is necessary in general, if you'd like to disable this step, please
    refer to this :ref:`example <sphx_glr_examples_80_extending_example_extending_data_preprocessor.py>`.

.. collapse:: <b>How can I turn off feature preprocessing?</b>

    Feature preprocessing is a single transformer which implements for example feature
    selection or transformation of features into a different space (i.e. PCA).

    This can be turned off by setting
    ``include={'feature_preprocessor'=["no_preprocessing"]}`` as shown in the example above.

.. collapse:: <b>Will non-scikit-learn models be added to Auto-sklearn?</b>

    The short answer: no.

    The long answer answer is a bit more nuanced: maintaining Auto-sklearn requires a lot of time and
    effort, which would grow even larger when depending on more libraries. Also, adding more
    libraries would require us to generate meta-data more often. Lastly, having more choices does not
    guarantee a better performance for most users as having more choices demands a longer search for
    good models and can lead to more overfitting.

    Nevertheless, everyone can still add their favorite model to Auto-sklearn's search space by
    following the `examples on how to extend Auto-sklearn
    <https://automl.github.io/auto-sklearn/master/examples/index.html#extension-examples>`_.

    If there is interest in creating a Auto-sklearn-contrib repository with 3rd-party models please
    open an issue for that.

.. collapse:: <b>How can I only search for interpretable models</b>

    Auto-sklearn can be restricted to only use interpretable models and preprocessing algorithms.
    Please see the Section :ref:`space` to learn how to restrict the models
    which are searched over or see the Example
    :ref:`sphx_glr_examples_40_advanced_example_interpretable_models.py`.

    We don't provide a judgement which of the models are interpretable as this is very much up to the
    specific use case, but would like to note that decision trees and linear models usually most
    interpretable.

Ensembling
==========

.. collapse:: <b>What can I configure wrt the ensemble building process?</b>

    The following hyperparameters control how the ensemble is constructed:

    * ``ensemble_class`` class object implementing :class:`autosklearn.ensembles.AbstractEnsemble`,
      will be instantiated by *auto-sklearn*'s ensemble builder.
    * ``ensemble_kwargs`` are keyword arguments that are passed to the ``ensemble_class`` upon
      instantiation. See below for an example argument.
    * ``ensemble_nbest`` allows the user to directly specify the number of models considered for the ensemble.  This hyperparameter can be an integer *n*, such that only the best *n* models are used in the final ensemble. If a float between 0.0 and 1.0 is provided, ``ensemble_nbest`` would be interpreted as a fraction suggesting the percentage of models to use in the ensemble building process (namely, if ensemble_nbest is a float, library pruning is implemented as described in `Caruana et al. (2006) <https://dl.acm.org/doi/10.1109/ICDM.2006.76>`_).
    * ``max_models_on_disc`` defines the maximum number of models that are kept on the disc, as a mechanism to control the amount of disc space consumed by *auto-sklearn*. Throughout the automl process, different individual models are optimized, and their predictions (and other metadata) is stored on disc. The user can set the upper bound on how many models are acceptable to keep on disc, yet this variable takes priority in the definition of the number of models used by the ensemble builder (that is, the minimum of ``ensemble_size``, ``ensemble_nbest`` and ``max_models_on_disc`` determines the maximal amount of models used in the ensemble). If set to None, this feature is disabled.

    The default method for Auto-sklearn is :class:`autosklearn.ensembles.EnsembleSelection`,
    which features the argument ``ensemble_size``. that determines the maximal size of the
    ensemble. Models can be added repeatedly, so the number of different models is usually
    less than the ``ensemble_size``.

.. collapse:: <b>Which models are in the final ensemble?</b>

    The results obtained from the final ensemble can be printed by calling ``show_models()`` or  ``leaderboard()``.
    The *auto-sklearn* ensemble is composed of scikit-learn models that can be inspected as exemplified
    in the Example :ref:`sphx_glr_examples_40_advanced_example_get_pipeline_components.py`.

.. collapse:: <b>Can I fit an ensemble also only post-hoc?</b>

    It is possible to build ensembles post-hoc. An example on how to do this (first searching for individual models, and then building an ensemble from them) can be seen in :ref:`sphx_glr_examples_60_search_example_sequential.py`.

Configuring the Search Procedure
================================

.. collapse:: <b>Can I change the resampling strategy?</b>

    Examples for using holdout and cross-validation can be found in :ref:`example <sphx_glr_examples_40_advanced_example_resampling.py>`

    If using a custom resampling strategy with predefined splits, you may need to disable
    the subsampling performed with particularly large datasets or if using a small ``memory_limit``.
    Please see the manual section on :ref:`limits`
    :class:`AutoSklearnClassifier(dataset_compression=...) <autosklearn.classification.AutoSklearnClassifier>`.
    for more details.

.. collapse:: <b>Can I use a custom metric</b>

    Examples for using a custom metric can be found in :ref:`example <sphx_glr_examples_40_advanced_example_metrics.py>`

Meta-Learning
=============

.. collapse:: <b>Which datasets are used for meta-learning?</b>

    We updated the list of datasets used for meta-learning several times and this list now differs
    significantly from the original 140 datasets we used in 2015 when the paper and the package were
    released. An up-to-date list of `OpenML task IDs <https://docs.openml.org/#tasks>`_ can be found
    on `github <https://github.com/automl/auto-sklearn/blob/master/scripts/update_metadata_util.py>`_.

.. collapse:: <b>How can datasets from the meta-data be excluded?</b>

    For *Auto-sklearn 1.0* one can pass the dataset name via the ``fit()`` function. If a dataset
    with the same name is within the meta-data, that datasets will not be used.

    For *Auto-sklearn 2.0* it is not possible to do so because of the method used to construct the
    meta-data.

.. collapse:: <b>Which meta-features are used for meta-learning?</b>

    We do not have a user guide on meta-features but they are all pretty simple and can be found
    `in the source code <https://github.com/automl/auto-sklearn/blob/master/autosklearn/metalearning/metafeatures/metafeatures.py>`_.

.. collapse:: <b>How is the meta-data generated for Auto-sklearn 1.0?</b>

    We currently generate meta-data the following way. First, for each of the datasets mentioned
    above, we run Auto-sklearn without meta-learning for a total of two days on multiple metrics (for
    classification these are accuracy, balanced accuracy, log loss and the area under the curce).
    Second, for each run we then have a look at each models that improved the score, i.e. the
    trajectory of the best known model at a time, and refit it on the whole training data. Third, for
    each of these models we then compute all scores we're interested in, these also include other
    ones such F1 and precision. Finally, for each combination of dataset and metric we store the best
    model we know of.

.. collapse:: <b>How is the meta-data generated for Auto-sklearn 2.0?</b>

    Please check `our paper <https://arxiv.org/abs/2007.04074>`_ for details.


Issues and Debugging
====================

.. collapse:: <b>How can I limit the number of model evaluations for debugging?</b>

    In certain cases, for example for debugging, it can be helpful to limit the number of
    model evaluations. We do not provide this as an argument in the API as we believe that it
    should NOT be used in practice, but that the user should rather provide time limits.
    An example on how to add the number of models to try as an additional stopping condition
    can be found `in this github issue <https://github.com/automl/auto-sklearn/issues/451#issuecomment-376445607>`_.
    Please note that Auto-sklearn will stop when either the time limit or the number of
    models termination condition is reached.

.. collapse:: <b>Why does the final ensemble contains only a dummy model?</b>

    This is a symptom of the problem that all runs started by Auto-sklearn failed. Usually, the issue
    is that the runtime or memory limit were too tight. Please check the output of
    ``sprint_statistics()`` to see the distribution of why runs failed. If there are mostly crashed
    runs, please check the log file for further details. If there are mostly runs that exceed the
    memory or time limit, please increase the respective limit and rerun the optimization.

.. collapse:: <b>Auto-sklearn does not use the specified amount of resources?</b>

    Auto-sklearn wraps scikit-learn and therefore inherits its parallelism implementation. In short,
    scikit-learn uses two modes of parallelizing computations:

    1. By using joblib to distribute independent function calls on multiple cores.
    2. By using lower level libraries such as OpenMP and numpy to distribute more fine-grained
       computation.

    This means that Auto-sklearn can use more resources than expected by the user. For technical
    reasons we can only control the 1st way of parallel execution, but not the 2nd. Thus, the user
    needs to make sure that the lower level parallelization libraries only use as many cores as
    allocated (on a laptop or workstation running a single copy of Auto-sklearn it can be fine to not
    adjust this, but when using a compute cluster it is necessary to align the parallelism setting
    with the number of requested CPUs). This can be done by setting the following environment
    variables: ``MKL_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``, ``BLIS_NUM_THREADS`` and
    ``OMP_NUM_THREADS``.

    More details can be found in the `scikit-learn docs <https://scikit-learn.org/stable/computing/parallelism.html?highlight=joblib#parallelism>`_.

Other
=====

.. collapse:: <b>Model persistence</b>

    *auto-sklearn* is mostly a wrapper around scikit-learn. Therefore, it is
    possible to follow the
    `persistence Example <https://scikit-learn.org/stable/modules/model_persistence.html>`_
    from scikit-learn.

.. collapse:: <b>Vanilla auto-sklearn</b>

    In order to obtain *vanilla auto-sklearn* as used in `Efficient and Robust Automated Machine Learning
    <https://papers.neurips.cc/paper/5872-efficient-and-robust-automated-machine-learning>`_
    set ``ensemble_class=autosklearn.ensembles.SingleBest`` and ``initial_configurations_via_metalearning=0``:

    .. code:: python

        import autosklearn.classification
        import autosklearn.ensembles
        automl = autosklearn.classification.AutoSklearnClassifier(
            ensemble_class=autosklearn.ensembles.SingleBest,
            initial_configurations_via_metalearning=0
        )

    This will always choose the best model according to the validation set.
    Setting the initial configurations found by meta-learning to zero makes
    *auto-sklearn* use the regular SMAC algorithm for suggesting new
    hyperparameter configurations.
