:orphan:

.. _manual:

Manual
~~~~~~

This manual shows how to use several aspects of auto-sklearn. It either
references the examples where possible or explains certain configurations.

Restrict Searchspace
*********************

Instead of using all available estimators, it is possible to restrict
*auto-sklearn*'s searchspace. The following shows an example of how to exclude
all preprocessing methods and restrict the configuration space to only
random forests.

>>> import autosklearn.classification
>>> automl = autosklearn.classification.AutoSklearnClassifier(include_estimators=["random_forest", ],
>>>                            exclude_estimators=None, include_preprocessors=["no_preprocessing", ],
>>>                            exclude_preprocessors=None)
>>> cls.fit(X_train, y_train)
>>> predictions = cls.predict(X_test, y_test)

**Note:** The strings used to identify estimators and preprocessors are the filenames without *.py*.

For a full list please have a look at the source code (in `autosklearn/pipeline/components/`):

  * `Classifiers <https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/classification>`_
  * `Regressors <https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/regression>`_
  * `Preprocessors <https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/feature_preprocessing>`_

Resampling strategies
*********************

Examples for using holdout and cross-validation can be found in `auto-sklearn/examples/ <https://github.com/automl/auto-sklearn/tree/master/example>`_

Parallel computation
********************

*auto-sklearn* supports parallel execution by data sharing on a shared file
system. In this mode, the SMAC algorithm shares the training data for it's
model by writing it to disk after every iteration. At the beginning of each
iteration, SMAC loads all newly found data points. An example can be found in
the example directory.

In it's default mode, *auto-sklearn* already uses two cores. The first one is
used for model building, the second for building an ensemble every time a new
machine learning model has finished training. The file `example_sequential
.py` in the example directory describes how to run these tasks sequentially
to use only a single core at a time.

Furthermore, depending on the installation of scikit-learn and numpy,
the model building procedure may use up to all cores. Such behaviour is
unintended by *auto-sklearn* and is most likely due to numpy being installed
from `pypi` as a binary wheel (`see here <http://scikit-learn-general.narkive
.com/44ywvAHA/binary-wheel-packages-for-linux-are-coming>`_). Executing
``export OPENBLAS_NUM_THREADS=1`` should disable such behaviours and make numpy
only use a single core at a time.

Model persistence
*****************

*auto-sklearn* is mostly a wrapper around scikit-learn. Therefore, it is
possible to follow the `persistence example
<http://scikit-learn.org/stable/modules/model_persistence.html#persistence-example>`_
from scikit-learn.