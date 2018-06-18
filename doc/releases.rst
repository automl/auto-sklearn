:orphan:

.. _releases:

..
    The following command allows to retrieve all commiters since a specified
    commit. From http://stackoverflow.com/questions/6482436/list-of-authors-in-git-since-a-given-commit
    git log 2e29eba.. --format="%aN <%aE>" --reverse | perl -e 'my %dedupe; while (<STDIN>) { print unless $dedupe{$_}++}'

========
Releases
========

Version 0.4.0
=============

* Fixes `#409 <https://github.com/automl/auto-sklearn/issues/409>`_: fixes
  ``predict_proba`` to no longer raise an `AttributeError`.
* Improved documentation of the parallel example.
* Classifiers are now tested to be idempotent as `required by scikit-learn
  <http://scikit-learn.org/stable/developers/contributing.html#estimated-attributes>`_.
* Fixes the usage of the shrinkage parameter in LDA.
* Fixes `#410 <https://github.com/automl/auto-sklearn/issues/410>`_ and changes
  the SGD hyperparameters
* Fixes `#425 <https://github.com/automl/auto-sklearn/issues/425>`_ which
  caused the non-linear support vector machine to always crash on OSX.
* Implements `#149 <https://github.com/automl/auto-sklearn/issues/149>`_: it
  is now possible to pass a custom cross-validation split following
  scikit-learn's ``model_selection`` module.
* It is now possible to decide whether or not to shuffle the data in
  Auto-sklearn by passing a bool `shuffle` in the dictionary of
  ``resampling_strategy_arguments``.
* Added functionality to track the test performance over time.
* Re-factored the ensemble building to be faster, read less data from the
  hard drive and perform random tie breaking in case of equally
  well-performing models.
* Implements `#438 <https://github.com/automl/auto-sklearn/issues/438>`_: To
  be consistent with the output of SMAC (which minimizes the loss of a target
  function), the output of the ensemble builder is now also the output of a
  minimization problem.
* Implements `#271 <https://github.com/automl/auto-sklearn/issues/271>`_:
  XGBoost is available again, even configuring the new dropout functionality.
* New documentation section `inspecting the results <http://automl.github.io/auto-sklearn/stable/manual.html#inspecting-the-results>`_.
* Fixes `#444 <https://github.com/automl/auto-sklearn/issues/444>`_:
  Auto-sklearn now only loads models for refit which are actually relevant
  for the ensemble.
* Adds an operating system check at import and installation time to make sure
  to not accidentaly run on a Windows machine.
* New examples gallery using sphinx gallery: `http://automl.github.io/auto-sklearn/stable/examples/index.html <http://automl.github.io/auto-sklearn/stable/examples/index.html>`_
* Safeguard Auto-sklearn against deleting directories it did not create (Issue
  `#317 <https://github.com/automl/auto-sklearn/issues/317>`_.

Contributors
************

* Matthias Feurer
* kaa
* Josh Mabry
* Katharina Eggensperger
* Vladimir Glazachev
* Jesper van Engelen
* Jin Woo Ahn
* Enrico Testa
* Marius Lindauer
* Yassine Morakakam

Version 0.3.0
=============

* Upgrade to scikit-learn 0.19.1.
* Do not use the ``DummyClassifier`` or ``DummyRegressor`` as part of an
  ensemble. Fixes `#140 <https://github.com/automl/auto-sklearn/issues/140>`_.
* Fixes #295 by loading the data in the subprocess instead of the main process.
* Fixes #326: refitting could result in a type error. This is now fixed by
  better type checking in the classification components.
* Updated search space for ``RandomForestClassifier``, ``ExtraTreesClassifier``
  and ``GradientBoostingClassifier`` (fixes #358).
* Removal of constant features is now a part of the pipeline.
* Allow passing an SMBO object into the ``AutoSklearnClassifier`` and
  ``AutoSklearnRegressor``.

Contributors
************

* Matthias Feurer
* Jesper van Engelen

Version 0.2.1
=============

* Allows the usage of scikit-learn 0.18.2.
* Upgrade to latest SMAC version (``0.6.0``) and latest random forest version
  (``0.6.1``).
* Added a Dockerfile.
* Added the possibility to change the size of the holdout set when
  using holdout resampling strategy.
* Fixed a bug in QDA's hyperparameters.
* Typo fixes in print statements.
* New method to retrieve the models used in the final ensemble.

Contributors
************

* Matthias Feurer
* Katharina Eggensperger
* Felix Leung
* caoyi0905
* Young Ryul Bae
* Vicente Alencar
* Lukas Großberger

Version 0.2.0
=============

* **auto-sklearn supports custom metrics and all metrics included in
  scikit-learn**. Different metrics can now be passed to the ``fit()``-method
  estimator objects, for example
  ``AutoSklearnClassifier.fit(metric='roc_auc')``.
* Upgrade to scikit-learn 0.18.1.
* Drop XGBoost as the latest release (0.6a2) does not work when spawned by
  the pyninsher.
* *auto-sklearn* can use multiprocessing in calls to ``predict()`` and
  ``predict_proba``. By `Laurent Sorber <https://github.com/lsorber>`_.

Contributors
************

* Matthias Feurer
* Katharina Eggensperger
* Laurent Sorber
* Rafael Calsaverini

Version 0.1.x
=============

There are no release notes for auto-sklearn prior to version 0.2.0.

Contributors
************

* Matthias Feurer
* Katharina Eggensperger
* Aaron Klein
* Jost Tobias Springenberg
* Anatolii Domashnev
* Stefan Falkner
* Alexander Sapronov
* Manuel Blum
* Diego Kobylkin
* Jaidev Deshpande
* Jongheon Jeong
* Hector Mendoza
* Timothy J Laurent
* Marius Lindauer
* _329_
* Iver Jordal
