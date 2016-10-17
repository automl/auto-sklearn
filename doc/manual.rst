:orphan:

.. _manual:

Manual
~~~~~~

This manual shows how to use several aspects of auto-sklearn. It either
references the examples where possible or explains certain configurations.

Resampling strategies
*********************

Examples for using holdout and cross-validation can be found in the example
directory of auto-sklearn.

Parallel computation
********************

auto-sklearn supports parallel execution by data sharing on a shared file
system. In this mode, the SMAC algorithm shares the training data for it's
model by writing it to disk after every iteration. At the beginning of each
iteration, SMAC loads all newly found data points. An example can be found in
the example directory.

In it's default mode, auto-sklearn already uses two cores. The first one is
used for model building, the second for building an ensemble every time a new
machine learning model has finished training. The file `example_sequential
.py` in the example directory describes how to run these tasks sequentially
to use only a single core at a time.

Furthermore, depending on the installation of scikit-learn and numpy,
the model building procedure may use up to all cores. Such behaviour is
unintended by auto-sklearn and is most likely due to numpy being installed
from `pypi` as a binary wheel (`see here http://scikit-learn-general.narkive
.com/44ywvAHA/binary-wheel-packages-for-linux-are-coming`_). Executing
``export OPENBLAS_NUM_THREADS=1`` should disable such behaviours and make numpy
only use a single core at a time.

Model persistence
*****************

auto-sklearn is mostly a wrapper around scikit-learn. Therefore, it is
possible to follow the `persistence example
http://scikit-learn.org/stable/modules/model_persistence.html#persistence-example`_
from scikit-learn.