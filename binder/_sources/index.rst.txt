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

.. topic:: NEW: Auto-sklearn 2.0

    Auto-sklearn 2.0 includes latest research on automatically configuring the AutoML system itself
    and contains a multitude of improvements which speed up the fitting the AutoML system.

*auto-sklearn 2.0* works the same way as regular *auto-sklearn* and you can use it via

    >>> from autosklearn.experimental.askl2 import AutoSklearn2Classifier

A paper describing our advances is available on `arXiv <https://arxiv.org/abs/2007.04074>`_.

Example
*******

    >>> import autosklearn.classification
    >>> import sklearn.model_selection
    >>> import sklearn.datasets
    >>> import sklearn.metrics
    >>> if __name__ == "__main__":
    >>>     X, y = sklearn.datasets.load_digits(return_X_y=True)
    >>>     X_train, X_test, y_train, y_test = \
                sklearn.model_selection.train_test_split(X, y, random_state=1)
    >>>     automl = autosklearn.classification.AutoSklearnClassifier()
    >>>     automl.fit(X_train, y_train)
    >>>     y_hat = automl.predict(X_test)
    >>>     print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))


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

If you are using Auto-sklearn 2.0, please also cite


 Auto-Sklearn 2.0: The Next Generation, Feurer *et al.*, to appear (2020).

 Bibtex entry::

     @article{ASKL2,
        title = {Auto-Sklearn 2.0},
        author = {Feurer, Matthias and Eggensperger, Katharina and
                  Falkner, Stefan and Lindauer, Marius and Hutter, Frank},
        booktitle = {Advances in Neural Information Processing Systems 28},
        year = {2020},
        journal = {arXiv:2006.???? [cs.LG]},
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
