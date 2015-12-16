*********************
What is auto-sklearn?
*********************

.. role:: bash(code)
    :language: bash

.. role:: python(code)
    :language: python

*auto-sklearn* is an automated machine learning toolkit and a drop-in
replacement for a scikit-learn estimator:

    >>> import autosklearn.classification
    >>> cls = autosklearn.classification.AutoSklearnClassifier()
    >>> cls.fit(X_train, y_train)
    >>> predictions = cls.predict(X_test, y_test)

*auto-sklearn* frees a machine learning user from algorithm selection and
hyperparameter tuning. It leverages recent advantages in *Bayesian
optimization*, *meta-learning* and *ensemble construction*. Learn more about
the technology behind *auto-sklearn* by reading a paper we just published at
the `AutoML workshop@ICML 2015 <https://sites.google.com/site/automlwsicml15/>`_
.

Example
*******

    >>> import autosklearn.classification
    >>> import sklearn.datasets
    >>> digits = sklearn.datasets.load_digits()
    >>> X = digits.data
    >>> y = digits.target
    >>> import numpy as np
    >>> indices = np.arange(X.shape[0])
    >>> np.random.shuffle(indices)
    >>> X = X[indices]
    >>> y = y[indices]
    >>> X_train = X[:1000]
    >>> y_train = y[:1000]
    >>> X_test = X[1000:]
    >>> y_test = y[1000:]
    >>> automl = autosklearn.classification.AutoSklearnClassifier()
    >>> automl.fit(X_train, y_train)
    >>> print(automl.score(X_test,y_test))


This will run for one hour should result in an accuracy above 0.98.


Installation
************
**Prerequisities**: *auto-sklearn* is written in python (2.7) and was developed
with Ubuntu. It should run on other Linux distributions, but won't work on a MAC
or on a windows PC. It requires scikit-learn 0.16.1, which in turn requires
numpy and scipy.

*auto-sklearn* has a dependency, which are not yet automatically resolved:

* `HPOlibConfigSpace <https://github.com/automl/HPOlibConfigSpace>`_

Please install these manually with:

.. code:: bash

    pip install -r https://raw.githubusercontent.com/automl/auto-sklearn/master/requ.txt

Then install *auto-sklearn*

.. code:: bash

    pip install git+https://github.com/automl/auto-sklearn.git#egg=autosklearn

We recommend installing *auto-sklearn* into a `virtual environment
<http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_ or into an
`anaconda environment <https://www.continuum.io/downloads>`_ because we have
seen strange things happening when installing it using
:bash:`python setup.py --user`.

API
***

.. autoclass:: autosklearn.classification.AutoSklearnClassifier


License
*******
*auto-sklearn* is licensed the same way as *scikit-learn*,
namely the 3-clause BSD license. The subprojects it uses, most notably SMAC,
may have different licenses.

Contributing
************
*auto-sklearn* is developed mainly by the `Machine Learning for Automated
Algorithm Design <http://aad.informatik.uni-freiburg.de>`_ group at the
University of Freiburg.

.. note::

    To avoid spending time on duplicate work or features that are unlikely to
    get merged, it is highly advised that you contact the developers
    by opening a `github issue <https://github
    .com/automl/auto-sklearn/issues>`_ before starting to work.

When developing new features, please create a new branch from the development
branch. Prior to submitting a pull request, make sure that all tests are
still passing.

Features under development
--------------------------
* support for arff files
* support for scikit-learn 0.17
* python 3 compability
* command line interface

A short guide to the code
-------------------------
* `automl.py`: main class which controls the workflow.
* `estimators.py`: wraps a scikit-learn interface around automl.py.
* `cli`: command line interface to the machine learning algorithm which is
  used internally by SMAC.
* `data`: code to read and store machine learning datasets.
* `evaluation`: classes to control the execution of machine learning,
  algorithms and resampling of the data.
* `metrics`: contains metrics which can be optimized.
* `util`: several utilityy functions.

Contributors
************

* Matthias Feurer
* Katharina Eggensperger
* Aaron Klein
* Jost Tobias Springenberg
* Manuel Blum
* Stefan Falkner
* Farooq Ahmed Zuberi
* Frank Hutter
* Alexander Sapronov

..
    Welcome to AutoSklearn's documentation!
    =======================================

    Contents:

    .. toctree::
        :maxdepth: 2

        Indices and tables
        ==================

        * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`
