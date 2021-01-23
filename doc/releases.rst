:orphan:

.. _releases:

..
    The following command allows to retrieve all commiters since a specified
    commit. From http://stackoverflow.com/questions/6482436/list-of-authors-in-git-since-a-given-commit
    git log 2e29eba.. --format="%aN <%aE>" --reverse | perl -e 'my %dedupe; while (<STDIN>) { print unless $dedupe{$_}++}'


========
Releases
========

Version 0.12.1
==============

* ADD: A new heuristic which gives a warning and subsamples the data if it is too large for the
  given ``memory_limit``.
* ADD #1024: Tune scikit-learn's ``MLPClassifier`` and ``MLPRegressor``.
* MAINT #1017: Improve the logging server introduced in release 0.12.0.
* MAINT #1024: Move to scikit-learn 0.24.X.
* MAINT #1038: Use new datasets for regression and classification and also update the metadata
  used for Auto-sklearn 1.0.
* MAINT #1040: Minor speed improvements in the ensemble selection algorithm.

Contributors v0.12.1
********************

* Matthias Feurer
* Katharina Eggensperger
* Francisco Rivera

Version 0.12.0
==============

* BREAKING: Auto-sklearn must now be guarded by ``__name__ == "__main__"`` due to the use of the
  ``spawn`` multiprocessing context.
* ADD #1026: Adds improved meta-data for Auto-sklearn 2.0 which results in strong improved
  performance.
* MAINT #984 and #1008: Move to scikit-learn 0.23.X
* MAINT #1004: Move from travis-ci to github actions.
* MAINT 8b67af6: drop the requirement to the lockfile package.
* FIX #990: Fixes a bug that made Auto-sklearn fail if there are missing values in a pandas
  DataFrame.
* FIX #1007, #1012 and #1014: Log multiprocessing output via a new log server. Remove several
  potential deadlocks related to the joint use of multi-processing, multi-threading and logging.

Contributors v0.12.0
********************

* Matthias Feurer
* ROHIT AGARWAL
* Francisco Rivera

Version 0.11.1
==============

* FIX #989: Fixes a bug where `y` was not passed to all data preprocessors which made 3rd party
  category encoders fail.
* FIX #1001: Fixes a bug which could make Auto-sklearn fail at random.
* MAINT #1000: Introduce a minimal version for ``dask.distributed``.

Contributors v0.11.1
********************

* Matthias Feurer

Version 0.11.0
==============

* ADD #992: Move ensemble building from being a separate process to a job submitted to the dask
  cluster. This allows for better control of the memory used in multiprocessing settings.
* FIX #905: Make ``AutoSklearn2Classifier`` picklable.
* FIX #970: Fix a bug where Auto-sklearn would fail if categorical features are passed as a
  Pandas Dataframe.
* MAINT #772: Improve error message in case of dummy prediction failure.
* MAINT #948: Finally use Pandas >= 1.0.
* MAINT #973: Improve meta-data by running meta-data generation for more time and separately for
  important metrics.
* MAINT #997: Improve memory handling in the ensemble building process. This allows building
  ensembles for larger datasets.

Contributors v0.11.0
********************

* Matthias Feurer
* Francisco Rivera
* Karl Leswing
* ROHIT AGARWAL

Version 0.10.0
==============

* ADD #325: Allow to separately optimize metrics for metadata generation.
* ADD #946: New dask backend for parallel Auto-sklearn.
* BREAKING #947: Drop Python3.5 support.
* BREAKING #946: Remove shared model mode for parallel Auto-sklearn.
* FIX #351: No longer pass un-picklable logger instances to the target function.
* FIX #840: Fixes a bug which prevented computing metadata for regression datasets. Also
  adds a unit test for regression metadata computation.
* FIX #897: Allow custom splitters to be used with multi-ouput regression.
* FIX #951: Fixes a lot of bugs in the regression pipeline that caused bad performance for
  regression datasets.
* FIX #953: Re-add `liac-arff` as a dependency.
* FIX #956: Fixes a bug which could cause Auto-sklearn not to find a model on disk which
  is part of the ensemble.
* FIX #961: Fixes a bug which caused Auto-sklearn to load bad meta-data for metrics which cannot
  be computed on multiclass datasets (especially ROC_AUC).
* DOC #498: Improve the example on resampling strategies by showing how to pass scikit-learn's
  splitter objects to Auto-sklearn.
* DOC #670: Demonstrate how to give access to training accuracy.
* DOC #872: Improve an example on how obtain the best model.
* DOC #940: Improve documentation of the docker image.
* MAINT: Improve the docker file by setting environment variable that restrict BLAS and OMP to only
  use a single core.
* MAINT #949: Replace `pip` by `pip3` in the installation guidelines.
* MAINT #280, #535, #956: Update meta-data and include regression meta-data again.

Contributors v0.10.0
********************

* Francisco Rivera
* Matthias Feurer
* felixleungsc
* Chu-Cheng Fu
* Francois Berenger

Version 0.9.0
=============

* ADD #157,#889: Improve handling of pandas dataframes, including the possibility to use pandas'
  categorical column type.
* ADD #375: New `SelectRates` feature preprocessing component for regression.
* ADD #891: Improve the robustness of Auto-sklearn by using the single best model if no ensemble
  is found.
* ADD #902: Track performance of the ensemble over time.
* ADD #914: Add an example on using pandas dataframes as input to Auto-sklearn.
* ADD #919: Add an example for multilabel classification.
* MAINT #909: Fix broken links in the documentation.
* MAINT #907,#911: Add initial support for mypy.
* MAINT #881,#927: Automatically build docker images on pushes to the master and development
  branch and also push them to dockerhub and the github docker registry.
* MAINT #918: Remove old dependencies from requirements.txt.
* MAINT #931: Add information about the host system and installed packages to the log file.
* MAINT #933: Reduce the number of warnings raised when building the documentation by sphinx.
* MAINT #936: Completely restructure the examples section.
* FIX #558: Provide better error message when the ensemble process fails due to a memory issue.
* FIX #901: Allow custom resampling strategies again (was broken due to an upgrade of SMAC).
* FIX #916: Fixes a bug where the data preprocessing configurations were ignored.
* FIX #925: make internal data preprocessing objects clonable.

Contributors v0.9.0
*******************

* Francisco Rivera
* Matthias Feurer
* felixleungsc
* Vladislav Skripniuk

Version 0.8
===========

* ADD #803: multi-output regression
* ADD #893: new Auto-sklearn mode Auto-sklearn 2.0

Contributors v0.8.0
*******************

* Chu-Cheng Fu
* Matthias Feurer

Version 0.7.1
=============

* ADD #764: support for automatic per_run_time_limit selection
* ADD #864: add the possibility to predict with cross-validation
* ADD #874: support to limit the disk space consumption
* MAINT #862: improved documentation and render examples in web page
* MAINT #869: removal of competition data manager support
* MAINT #870: memory improvements when building ensemble
* MAINT #882: memory improvements when performing ensemble selection
* FIX #701: scaling factors for metafeatures should not be learned using test data
* FIX #715: allow unlimited ML memory
* FIX #771: improved worst possible result calculation
* FIX #843: default value for SelectPercentileRegression
* FIX #852: clip probabilities within [0-1]
* FIX #854: improved tmp file naming
* FIX #863: SMAC exceptions also registered in log file
* FIX #876: allow Auto-sklearn model to be cloned
* FIX #879: allow 1-D binary predictions

Contributors v0.7.1
*******************

* Matthias Feurer
* Xiaodong DENG
* Francisco Rivera

Version 0.7.0
=============

* ADD #785: user control to reduce the hard drive memory required to store ensembles
* ADD #794: iterative fit for gradient boosting
* ADD #795: add successive halving evaluation strategy
* ADD #814: new sklearn.metrics.balanced_accuracy_score instead of custom metric
* ADD #815: new experimental evaluation mode called iterative_cv
* MAINT #774: move from scikit-learn 0.21.X to 0.22.X
* MAINT #791: move from smac 0.8 to 0.12
* MAINT #822: make autosklearn modules PEP8 compliant
* FIX #733: fix for n_jobs=-1
* FIX #739: remove unnecessary warning
* FIX ##769: fixed error in calculation of meta features
* FIX #778: support for python 3.8
* FIX #781: support for pandas 1.x

Contributors v0.7.0
*******************

* Andrew Nader
* Gui Miotto
* Julian Berman
* Katharina Eggensperger
* Matthias Feurer
* Maximilian Peters
* Rong-Inspur
* Valentin Geffrier
* Francisco Rivera

Version 0.6.0
=============

* MAINT: move from scikit-learn 0.19.X to 0.21.X
* MAINT #688: allow for pyrfr version 0.8.X
* FIX #680: Remove unnecessary print statement
* FIX #600: Remove unnecessary warning

Contributors v0.6.0
*******************

* Guilherme Miotto
* Matthias Feurer
* Jin Woo Ahn

Version 0.5.2
=============

* FIX #669: Correctly handle arguments to the ``AutoMLRegressor``
* FIX #667: Auto-sklearn works with numpy 1.16.3 again.
* ADD #676: Allow brackets [ ] inside the temporary and output directory paths.
* ADD #424: (Experimental) scripts to reproduce the results from the original Auto-sklearn paper.

Contributors v0.5.2
*******************

* Jin Woo Ahn
* Herilalaina Rakotoarison
* Matthias Feurer
* yazanobeidi

Version 0.5.1
=============

* ADD #650: Auto-sklearn will immediately stop if prediction using scikit-learn's dummy predictor
  fail.
* ADD #537: Auto-sklearn will no longer start for time limits less than 30 seconds.
* FIX #655: Fixes an issue where predictions using models from parallel Auto-sklearn runs could
  be wrong.
* FIX #648: Fixes an issue with custom meta-data directories.
* FIX #626: Fixes an issue where losses were not minimized, but maximized.
* MAINT #646: Do no longer restrict the numpy version to be less than 1.14.5.

Contributors v0.5.1
*******************

* Jin Woo Ahn
* Taneli Mielikäinen
* Matthias Feurer
* jianswang

Version 0.5.0
=============

* ADD #593: Auto-sklearn supports the ``n_jobs`` argument for parallel
  computing on a single machine.
* DOC #618: Added links to several system requirements.
* Fixes #611: Improved installation from pip.
* TEST #614: Test installation with clean Ubuntu on travis-ci.
* MAINT: Fixed broken link and typo in the documentation.

Contributors v0.5.0
*******************

* Mohd Shahril
* Adrian
* Matthias Feurer
* Jirka Borovec
* Pradeep Reddy Raamana


Version 0.4.2
=============

* Fixes #538: Remove rounding errors when giving a training set fraction for
  holdout.
* Fixes #558: Ensemble script now uses less memory and the memory limit can be
  given to Auto-sklearn.
* Fixes #585: Auto-sklearn's ensemble script produced wrong results when
  called directly (and not via one of Auto-sklearn's estimator classes).
* Fixes an error in the ensemble script which made it non-deterministic.
* MAINT #569: Rename hyperparameter to have a different name than a
  scikit-learn hyperparameter with different meaning.
* MAINT #592: backwards compatible requirements.txt
* MAINT #588: Fix SMAC version to 0.8.0
* MAINT: remove dependency on the six package
* MAINT: upgrade to XGBoost 0.80

Contributors v0.4.2
*******************

* Taneli Mielikäinen
* Matthias Feurer
* Diogo Bastos
* Zeyi Wen
* Teresa Conceição
* Jin Woo Ahn

Version 0.4.1
=============

* Added documentation on `how to extend Auto-sklearn <https://github.com/automl/auto-sklearn/pull/510>`_
  with custom classifier, regressor, and preprocessor.
* Auto-sklearn now requires numpy version between 1.9.0 and 1.14.5, due to higher versions
  causing travis failure.
* Examples now use ``sklearn.datasets.load_breast_cancer()`` instead of ``sklearn.datasets.load_digits()``
  to reduce memory usage for travis build.
* Fixes future warnings on non-tuple sequence for indexing.
* Fixes `#500 <https://github.com/automl/auto-sklearn/issues/500>`_: fixes
  ensemble builder to correctly evaluate model score with any metrics.
  See this `PR <https://github.com/automl/auto-sklearn/pull/522>`_.
* Fixes `#482 <https://github.com/automl/auto-sklearn/issues/482>`_ and
  `#491 <https://github.com/automl/auto-sklearn/issues/491>`_: Users can now set up
  custom logger configuration by passing a dictionary created by a yaml file to
  ``logging_config``.
* Fixes `#566 <https://github.com/automl/auto-sklearn/issues/566>`_: ensembles are now sorted correctly.
* Fixes `#293 <https://github.com/automl/auto-sklearn/issues/293>`_: Auto-sklearn checks if appropriate
  target type was given for classification and regression before call to ``fit()``.
* Travis-ci now runs flake8 to enforce pep8 style guide, and uses travis-ci instead of circle-ci
  for deployment.

Contributors v0.4.1
*******************

* Matthias Feurer
* Manuel Streuhofer
* Taneli Mielikäinen
* Katharina Eggensperger
* Jin Woo Ahn

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

Contributors v0.4.0
*******************

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

Contributors v0.3.0
*******************

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

Contributors v0.2.1
*******************

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

Contributors v0.2.0
*******************

* Matthias Feurer
* Katharina Eggensperger
* Laurent Sorber
* Rafael Calsaverini

Version 0.1.x
=============

There are no release notes for auto-sklearn prior to version 0.2.0.

Contributors v0.1.x
*******************

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
