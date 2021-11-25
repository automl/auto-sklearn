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
`NIPS 2015 <https://papers.neurips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf>`_
.

.. topic:: NEW: Auto-sklearn 2.0

    Auto-sklearn 2.0 includes latest research on automatically configuring the AutoML system itself
    and contains a multitude of improvements which speed up the fitting the AutoML system.

*auto-sklearn 2.0* works the same way as regular *auto-sklearn* and you can use it via

.. code:: python

    from autosklearn.experimental.askl2 import AutoSklearn2Classifier

A paper describing our advances is available on `arXiv <https://arxiv.org/abs/2007.04074>`_.

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


auto-sklearn acceleration with sklearnex
****************************************
You can accelerate auto-sklearn with `Intel(R) Extension for Scikit-Learn (sklearnex) <https://github.com/intel/scikit-learn-intelex>`_. The acceleration is achieved through `patching <https://intel.github.io/scikit-learn-intelex/what-is-patching.html>`_: replacing scikit-learn algorithms with their optimized versions provided by the extension.

Read `system requirements <https://intel.github.io/scikit-learn-intelex/system-requirements.html>`_ and install sklearnex with pip or conda:

.. code:: bash

    pip install scikit-learn-intelex

.. code:: bash

    conda install scikit-learn-intelex

.. code:: bash

    conda install -c conda-forge scikit-learn-intelex

To accelerate auto-sklearn, insert the following two lines of patching code before auto-sklearn and sklearn imports:

.. code:: python

    from sklearnex import patch_sklearn
    patch_sklearn()

    import autosklearn.classification

To return to the original scikit-learn implementation, unpatch scikit-learn and reimport auto-sklearn and sklearn:

.. code:: python

    from sklearnex import unpatch_sklearn
    unpatch_sklearn()

    import autosklearn.classification


Manual
******

* :ref:`installation`
* :ref:`manual`
* :ref:`api`
* :ref:`extending`
* :ref:`faq`


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
        author    = {Feurer, Matthias and Klein, Aaron and Eggensperger, Katharina  Springenberg, Jost and Blum, Manuel and Hutter, Frank},
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
