:orphan:

.. _faq:

===
FAQ
===

Issues
======

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

Log files and output
====================

.. collapse:: <b>Where does Auto-sklearn output files by default?</b>

    *Auto-sklearn* heavily uses the hard drive to store temporary data, models and log files which can
    be used to inspect the behavior of Auto-sklearn. Each run of Auto-sklearn requires
    its own directory. If not provided by the user, *Auto-sklearn* requests a temporary directory from
    Python, which by default is located under ``/tmp`` and starts with ``autosklearn_tmp_`` followed
    by a random string. By default, this directory is deleted when the *Auto-sklearn* object is
    destroyed. If you want to keep these files you can pass the argument
    ``delete_tmp_folder_after_terminate=True`` to the *Auto-sklearn* object.

    The :class:`autosklearn.classification.AutoSklearnClassifier` and all other *auto-sklearn*
    estimators accept the argument ``tmp_directory`` which change where such output is written to.

    There's an additional argument ``output_directory`` which can be passed to *Auto-sklearn* and it
    controls where test predictions of the ensemble are stored if the test set is passed to ``fit()``.

.. collapse:: <b>Auto-sklearn eats up all my disk space</b>

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

Available machine learning models
=================================

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

.. collapse:: <b>Can the preprocessing be disabled</b>

    Feature preprocessing can be disabled as discussed in the example
    :ref:`space`. Other preprocessing steps such as one hot encoding, missing
    feature imputation and normalization cannot yet be disabled, but we're working on that.

Usage
=====

.. collapse:: <b>Only use interpretable models</b>

    Auto-sklearn can be restricted to only use interpretable models and preprocessing algorithms.
    Please see the Section :ref:`space` to learn how to restrict the models
    which are searched over or see the Example
    :ref:`sphx_glr_examples_40_advanced_example_interpretable_models.py`.

    We don't provide a judgement which of the models are interpretable as this is very much up to the
    specific use case, but would like to note that decision trees and linear models usually most
    interpretable.

.. collapse:: <b>Limiting the number of model evaluations</b>

    In certain cases, for example for debugging, it can be helpful to limit the number of
    model evaluations. We do not provide this as an argument in the API as we believe that it
    should NOT be used in practice, but that the user should rather provide time limits.
    An example on how to add the number of models to try as an additional stopping condition
    can be found `in this github issue <https://github.com/automl/auto-sklearn/issues/451#issuecomment-376445607>`_.
    Please note that Auto-sklearn will stop when either the time limit or the number of
    models termination condition is reached.

.. collapse:: <b>Ensemble contains only a dummy model</b>

    This is a symptom of the problem that all runs started by Auto-sklearn failed. Usually, the issue
    is that the runtime or memory limit were too tight. Please check the output of
    ``sprint_statistics`` to see the distribution of why runs failed. If there are mostly crashed
    runs, please check the log file for further details. If there are mostly runs that exceed the
    memory or time limit, please increase the respective limit and rerun the optimization.

.. collapse:: <b>Parallel processing and oversubscription</b>

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
