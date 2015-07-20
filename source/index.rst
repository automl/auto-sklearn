*********************
What is auto-sklearn?
*********************

.. role:: bash(code)
    :language: bash

*auto-sklearn* is an automated machine learning toolkit and a drop-in
replacement for a scikit-learn estimator.

*auto-sklearn* frees a machine learning user from algorithm selection and
hyperparameter tuning. It leverages recent advantages in *Bayesian
optimization*, *meta-learning* and *ensemble construction*. Learn more about
the technology behind *auto-sklearn* by reading a paper we just published at
the `AutoML workshop@ICML 2015 <https://sites.google.com/site/automlwsicml15/>`_
.

Example
*******

    >>> import autosklearn
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
    >>> automl = autosklearn.AutoSklearnClassifier()
    >>> automl.fit(X_train, y_train)
    >>> print automl.score(X_test, y_test)
    
This will run for one hour should result in an accuracy above 0.99.

Installation
************
*auto-sklearn* has several dependencies, which are not yet automatically
resolved:

* `HPOlib <https://github.com/automl/HPOlib>`_
* `HPOlibConfigSpace <https://github.com/automl/HPOlibConfigSpace>`_
* `ParamSklearn <https://bitbucket.org/mfeurer/paramsklearn>`_
* `pyMetaLearn <https://bitbucket.org/mfeurer/pymetalearn>`_

Please install these manually, for example by:

.. code:: bash

    pip install scikit-learn==0.15.2
    pip install git+https://github.com/mfeurer/HPOlibConfigSpace#egg=HPOlibConfigSpace0.1dev
    pip install git+https://git@bitbucket.org/mfeurer/paramsklearn.git@73d8643b2849db753ddc7b8909d01e6cee9bafc6 --no-deps
    pip install git+https://github.com/automl/HPOlib#egg=HPOlib0.2
    pip install --editable git+https://bitbucket.org/mfeurer/pymetalearn/#egg=pyMetaLearn

Then install *auto-sklearn*

.. code:: bash

    git clone https://github.com/automl/auto-sklearn.git
    cd auto-sklearn
    python setup.py install

API
***

.. autoclass:: autosklearn.AutoSklearnClassifier


License
*******
*auto-sklearn* features the same license as *scikit-learn*,
namely the 3-clause BSD license. The subprojects it uses may have different
licenses.

Contributors
************

*auto-sklearn* is developed by the `Machine Learning for Automated Algorithm
Design <http://aad.informatik.uni-freiburg.de>`_ group at the University of
Freiburg. Contributors are:

* Matthias Feurer
* Katharina Eggensperger
* Aaron Klein
* Jost Tobias Springenberg
* Manuel Blum
* Stefan Falkner
* Farooq Ahmed Zuberi
* Frank Hutter

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
