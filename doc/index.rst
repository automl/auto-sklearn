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
the technology behind *auto-sklearn* by reading this paper published at
the `NIPS 2015 <http://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf>`_
.

Example
*******

    >>> import autosklearn.classification
    >>> import sklearn.cross_validation
    >>> import sklearn.datasets
    >>> import sklearn.metrics
    >>> digits = sklearn.datasets.load_digits()
    >>> X = digits.data
    >>> y = digits.target
    >>> X_train, X_test, y_train, y_test = \
            sklearn.cross_validation.train_test_split(X, y, random_state=1)
    >>> automl = autosklearn.classification.AutoSklearnClassifier()
    >>> automl.fit(X_train, y_train)
    >>> y_hat = automl.predict(X_test)
    >>> print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))


This will run for one hour should result in an accuracy above 0.98.


Installation
************
**Prerequisities**: *auto-sklearn* is written in python and was developed
with Ubuntu. It should run on other Linux distributions, but won't work on a MAC
or on a windows PC. We aim to always support the two latests python versions,
which are 3.4 and 3.5 at the moment. It is built around scikit-learn 0.17.1 and
needs a compiler for C++ 11.

Please install all dependencies manually with:

.. code:: bash

    curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install

Then install *auto-sklearn*

.. code:: bash

    pip install auto-sklearn

We recommend installing *auto-sklearn* into a `virtual environment
<http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.

Manual
******

* :ref:`API`
* :ref:`manual`
* :ref:`extending`


License
*******
*auto-sklearn* is licensed the same way as *scikit-learn*,
namely the 3-clause BSD license.

Citing auto-sklearn
*******************

If you use auto-sklearn in a scientific publication, we would appreciate
references to the following paper:


 `Efficient and Robust Automated Machine Learning
 <https://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning>`_,
 Feurer *et al.*, Advances in Neural Information Processing Systems 28 (NIPS 2015).

 Bibtex entry::

     @incollection{NIPS2015_5872,
        title = {Efficient and Robust Automated Machine Learning},
        author = {Feurer, Matthias and Klein, Aaron and Eggensperger, Katharina and
                  Springenberg, Jost and Blum, Manuel and Hutter, Frank},
        booktitle = {Advances in Neural Information Processing Systems 28},
        editor = {C. Cortes and N. D. Lawrence and D. D. Lee and M. Sugiyama and R. Garnett},
        pages = {2962--2970},
        year = {2015},
        publisher = {Curran Associates, Inc.},
        url = {http://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf}
     }

Contributing
************

We appreciate all contribution to auto-sklearn, from bug reports,
documentation to new features. If you want to contribute to the code, you can
pick an issue from the `issue tracker <https://github.com/automl/auto-sklearn/issues>`_
which is marked with `Needs contributer`.

.. note::

    To avoid spending time on duplicate work or features that are unlikely to
    get merged, it is highly advised that you contact the developers
    by opening a `github issue <https://github
    .com/automl/auto-sklearn/issues>`_ before starting to work.

When developing new features, please create a new branch from the development
branch. Prior to submitting a pull request, make sure that all tests are
still passing.

Contributors
************

* Matthias Feurer
* Katharina Eggensperger
* Jost Tobias Springenberg
* Aaron Klein
* Anatolii Domashnev
* Alexander Sapronov
* Stefan Falkner
* Manuel Blum
* Hector Mendoza
* Farooq Ahmed Zuberi
* Frank Hutter
* Diego Kobylkin
* Marius Lindauer


