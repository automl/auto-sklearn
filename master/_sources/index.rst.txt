************
auto-sklearn
************

.. role:: bash(code)
    :language: bash

.. role:: python(code)
    :language: python

*auto-sklearn* is an automated machine learning toolkit and a drop-in
replacement for a scikit-learn estimator:

    >>> import autosklearn.classification
    >>> cls = autosklearn.classification.AutoSklearnClassifier()
    >>> cls.fit(X_train, y_train)
    >>> predictions = cls.predict(X_test)

*auto-sklearn* frees a machine learning user from algorithm selection and
hyperparameter tuning. It leverages recent advantages in *Bayesian
optimization*, *meta-learning* and *ensemble construction*. Learn more about
the technology behind *auto-sklearn* by reading our paper published at
`NIPS 2015 <http://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf>`_
.

Example
*******

    >>> import autosklearn.classification
    >>> import sklearn.model_selection
    >>> import sklearn.datasets
    >>> import sklearn.metrics
    >>> X, y = sklearn.datasets.load_digits(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(X, y, random_state=1)
    >>> automl = autosklearn.classification.AutoSklearnClassifier()
    >>> automl.fit(X_train, y_train)
    >>> y_hat = automl.predict(X_test)
    >>> print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))


This will run for one hour and should result in an accuracy above 0.98.


Manual
******

* :ref:`installation`
* :ref:`manual`
* :ref:`api`
* :ref:`extending`


License
*******
*auto-sklearn* is licensed the same way as *scikit-learn*,
namely the 3-clause BSD license.

Citing auto-sklearn
*******************

If you use auto-sklearn in a scientific publication, we would appreciate a
reference to the following paper:


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

We appreciate all contribution to auto-sklearn, from bug reports and
documentation to new features. If you want to contribute to the code, you can
pick an issue from the `issue tracker <https://github.com/automl/auto-sklearn/issues>`_
which is marked with `Needs contributer`.

.. note::

    To avoid spending time on duplicate work or features that are unlikely to
    get merged, it is highly advised that you contact the developers
    by opening a `github issue <https://github
    .com/automl/auto-sklearn/issues>`_ before starting to work.

When developing new features, please create a new branch from the development
branch. When to submitting a pull request, make sure that all tests are
still passing.
