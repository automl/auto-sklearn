:orphan:

.. _manual:

===
FAQ
===

Errors
======

Auto-sklearn is extremely memory hungry in a sequential setting
---------------------------------------------------------------

Auto-sklearn can appear very memory hungry (i.e. requiring a lot of memory for small datasets) due
to the use of ``fork`` for creating new processes when running in sequential manner (if this
happens in a parallel setting or if you pass your own dask client this is due to a different
issue).

Let's go into some more detail and discuss how to fix it:
Auto-sklearn executes each machine learning algorithm in its own process to be able to apply a
memory limit and a time limit. To start such a process, Python gives three options: ``fork``,
``forkserver`` and ``spawn``. The default ``fork`` copies the whole process memory into the
subprocess. If the main process already uses 1.5GB of main memory and we apply a 3GB memory
limit to Auto-sklearn, it will only be able to use 1.5GB of that. We would have loved to use
``forkserver`` or ``spawn`` instead, which both don't suffer from this issue (and have some
further improvements, see `here <https://pythonspeed.com/articles/python-multiprocessing/>`_), but
they require the Auto-sklearn code to be guarded by ``if __name__ == "__main__"`` or executed in a
notebook and we decided that we do not want to require this by default.

There are now two possible solutions:

1. Use parallel Auto-sklearn: if you use Auto-sklean in parallel, it defaults to ``forkserver``
   as the parallelization mechanism itself requires Auto-sklearn the code to be guarded.
2. Pass a `dask client <https://distributed.dask.org/en/latest/client.html>`_. If the user passes
   a dask client, Auto-sklearn can no longer assume that it runs in sequential mode and will use
   a ``forkserver`` to start new processes.

We therefore suggest using one of the above settings by default.

Auto-sklearn is extremely memory hungry in a sequential setting
---------------------------------------------------------------

When running Auto-sklearn in a parallel setting it starts new processes for evaluating machine
learning models using the ``forkserver`` mechanism. If not all code in the main script is guarded
by ``if __name__ == "__main__"`` it is executed for each subprocess. If now part of the code that
is not guarded is the code that is loading your dataset, it is executed for every evaluation of a
machine learning algorithm, blocking your RAM.

We therefore suggest moving all code inside the main block or functions.

Auto-sklearn crashes with a segmentation fault
----------------------------------------------

Please make sure that you have read and followed the :ref:`installation` section! In case
everything is set up correctly, this is most likely due to the dependency
`pyrfr <https://github.com/automl/random_forest_run>`_ not being compiled correctly. If this is the
case please execute

.. code:: python

    import pyrfr.regression as reg
    data = reg.default_data_container(64)

If this fails, the pyrfr dependency is indeed not compiled correctly. We advice you to do the
following:

1. Check if you can use a pre-compiled version of the pyrfr to avoid compiling it yourself.
2. If you are not yet using Conda, consider using it.
3. Install correct build dependencies before installing the pyrfr, you can check the following
   github issues for suggestions: `1025 <https://github.com/automl/auto-sklearn/issues/1025>`_,
   `856 <https://github.com/automl/auto-sklearn/issues/856>`_

Log files and output
====================

Where does Auto-sklearn output files by default?
------------------------------------------------

*auto-sklearn* heavily uses the hard drive to store temporary data, models and log files which can
be used to inspect the behavior auto Auto-sklearn. Each run of Auto-sklearn requires
its own directory *auto-sklearn* requests a temporary directory from
Python, which by default is located under ``/tmp`` and starts with ``autosklearn_tmp_`` followed
by a random string.

The :class:`autosklearn.classification.AutoSklearnClassifier` and all other *auto-sklearn*
estimators accept the argument ``tmp_directory`` which change where such output is written to.

There's an additional argument ``output_directory`` which can be passed to *auto-sklearn* and it
controls where test predictions of the ensemble are stored if the test set is passed to ``fit()``.

Auto-sklearn eats up all my disk space
--------------------------------------

*auto-sklearn* heavily uses the hard drive to store temporary data, models and log files which can
be used to inspect the behavior auto Auto-sklearn. By default, *auto-sklearn* stores around 50
models and their predictions on the validation data (which is a subset of the training data in
case of holdout and the full training data in case of cross-validation) on the hard drive. Models
are removed everytime the ensemble builder finishes an iteration, which means that the number of
models stored on disk is only an approximation. One can therefore change the number of models
that will be stored on disk by passing an integer for the argument ``max_models_on_disc``.

As the number of models is only an indicator of the disk space used it is also possible to pass
the memory in MB the models are allowed to use as a float. As above this is rather a guideline on
how much memory is used as the models are removed from disk everytime the ensemble builder
finishes an iteration. Especially when running in parallel it can happen that multiple models are
constructed during one run of the ensemble builder.

Available machine learning models
=================================

Will non-scikit-learn models be added to Auto-sklearn?
------------------------------------------------------

The short answer is unfortunately no.

The long answer answer is a bit more nuanced: maintaining Auto-sklearn requires a lot of time and
effort, which would grow even larger when depending on more libraries. Also, adding more
libraries would require us to generate meta-data more often. Lastly, having more choices does not
guarantee a better performance for most users as having more choices demands a longer search for
good models and can lead to more overfitting.

Nevertheless, everyone can still add his or her favorite model to Auto-sklearn's search space by
following the `examples on how to extend Auto-sklearn
<https://automl.github.io/auto-sklearn/master/examples/index.html#extension-examples>`_.

If there is interest in creating a auto-sklearn-contrib repository with 3rd-party models please
open an issue for that.

Can the preprocessing be disabled
---------------------------------

No, but we're working on that.

Usage
=====

Only use interpretable models
-----------------------------

Auto-sklearn can be restricted to only use interpretable models and preprocessing algorithms.
Please see the section :ref:`Restricting the searchspace` to learn how to restrict the models
which are searched over or see
`this example <examples/40_advanced/example_interpretable_models.html>`_.

We don't provide a judgement which of the models are interpretable as this is very much up to the
specific use case, but would like to note that decision trees and linear models usually most
interpretable.

Passing the number of models to try
-----------------------------------

In certain cases, for example for debugging, it can be helpful to limit the number of
models to try. We do not provide this as an argument in the API as we believe that it
should NOT be used in practice, but that the user should rather provide time limits.
An example on how to add the number of models to try as an additional stopping condition
can be found `in this github issue <https://github.com/automl/auto-sklearn/issues/451#issuecomment-376445607>`_.
Please note that Auto-sklearn will stop when either the time limit or the number of
models termination condition is reached.

Ensemble contains only a dummy model
------------------------------------

TODO

3. How to set useful budgets: https://github.com/automl/auto-sklearn/issues/57

Parallel processing and oversubscription
----------------------------------------

TODO

Meta-Learning
=============

Which datasets are used for meta-learning?
------------------------------------------

We updated the list of datasets used for meta-learning several times and this list now differes
significantly from the original 140 datasets we used in 2015 when the paper and the package were
released. An up-to-date list of `OpenML task IDs <https://docs.openml.org/#tasks>`_ can be found
on `github <https://github.com/automl/auto-sklearn/blob/master/scripts/update_metadata_util.py>`_

Which meta-features are used for meta-learning?
-----------------------------------------------

We do not have a user guide on meta-features but they are all pretty simple and can be found
`in the source code <https://github.com/automl/auto-sklearn/blob/master/autosklearn/metalearning/metafeatures/metafeatures.py>`_.
