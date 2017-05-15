:orphan:

.. _releases:

..
    The following command allows to retrieve all commiters since a specified
    commit. From http://stackoverflow.com/questions/6482436/list-of-authors-in-git-since-a-given-commit
    git log 2e29eba.. --format="%aN <%aE>" --reverse | perl -e 'my %dedupe; while (<STDIN>) { print unless $dedupe{$_}++}'

========
Releases
========

Version 0.2.0
=============

Major changes
~~~~~~~~~~~~~

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
~~~~~~~~~~~~

* Matthias Feurer
* Katharina Eggensperger
* Laurent Sorber
* Rafael Calsaverini

Version 0.1.x
=============

There are no release notes for auto-sklearn prior to version 0.2.0.

Contributors
~~~~~~~~~~~~

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
