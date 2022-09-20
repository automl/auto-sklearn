************
auto-sklearn
************

.. role:: bash(code)
    :language: bash

.. role:: python(code)
    :language: python

*auto-sklearn* is an automated machine learning toolkit and a drop-in
replacement for a scikit-learn estimator:

.. code:: python

    import autosklearn.classification
    cls = autosklearn.classification.AutoSklearnClassifier()
    cls.fit(X_train, y_train)
    predictions = cls.predict(X_test)

*auto-sklearn* frees a machine learning user from algorithm selection and
hyperparameter tuning. It leverages recent advantages in *Bayesian
optimization*, *meta-learning* and *ensemble construction*. Learn more about
the technology behind *auto-sklearn* by reading our paper published at
`NeurIPS 2015 <https://papers.neurips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf>`_
.

.. topic:: NEW: Text feature support

    Auto-sklearn now supports text features, check our new example:
    :ref:`sphx_glr_examples_40_advanced_example_text_preprocessing.py`


Example
*******

.. code:: python

    import autosklearn.classification
    import sklearn.model_selection
    import sklearn.datasets
    import sklearn.metrics

    if __name__ == "__main__":
        X, y = sklearn.datasets.load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(X, y, random_state=1)
        automl = autosklearn.classification.AutoSklearnClassifier()
        automl.fit(X_train, y_train)
        y_hat = automl.predict(X_test)
        print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

This will run for one hour and should result in an accuracy above 0.98.

Manual
******

* :ref:`installation`
* :ref:`manual`
* :ref:`api`
* :ref:`extending`
* :ref:`faq`

Additional Material
*******************

We provide slides and notebooks from talks and tutorials here:
`auto-sklearn-talks <https://github.com/automl/auto-sklearn-talks>`_

License
*******
*auto-sklearn* is licensed the same way as *scikit-learn*,
namely the 3-clause BSD license.

Citing auto-sklearn
*******************

If you use auto-sklearn in a scientific publication, we would appreciate a
reference to the following paper:


 `Efficient and Robust Automated Machine Learning
 <https://papers.neurips.cc/paper/5872-efficient-and-robust-automated-machine-learning>`_,
 Feurer *et al.*, Advances in Neural Information Processing Systems 28 (NIPS 2015).

 Bibtex entry::

    @inproceedings{feurer-neurips15a,
        title     = {Efficient and Robust Automated Machine Learning},
        author    = {Feurer, Matthias and Klein, Aaron and Eggensperger, Katharina and Springenberg, Jost and Blum, Manuel and Hutter, Frank},
        booktitle = {Advances in Neural Information Processing Systems 28 (2015)},
        pages     = {2962--2970},
        year      = {2015}
    }

If you are using Auto-sklearn 2.0, please also cite


 `Auto-Sklearn 2.0: Hands-free AutoML via Meta-Learning <https://arxiv.org/abs/2007.04074>`_, Feurer *et al.*, (arXiv, 2020).

 Bibtex entry::

    @article{feurer-arxiv20a,
        title     = {Auto-Sklearn 2.0: Hands-free AutoML via Meta-Learning},
        author    = {Feurer, Matthias and Eggensperger, Katharina and Falkner, Stefan and Lindauer, Marius and Hutter, Frank},
        journal   = {arXiv:2007.04074 [cs.LG]},
        year      = {2020},
    }

Contributing
************

We appreciate all contribution to auto-sklearn, from bug reports and
documentation to new features. If you want to contribute to the code, you can
pick an issue from the `issue tracker <https://github.com/automl/auto-sklearn/issues>`_.

Check out our `contribution guide on github <https://github.com/automl/auto-sklearn/blob/master/CONTRIBUTING.md>`_ if you want to know more!
We've catered it for both new and experienced contributers.

.. note::

    To avoid spending time on duplicate work or features that are unlikely to
    get merged, it is highly advised that you contact the developers
    by opening a `github issue <https://github
    .com/automl/auto-sklearn/issues>`_ before starting to work.
