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

Model persistence
*****************

auto-sklearn is mostly a wrapper around scikit-learn. Therefore, it is
possible to follow the `persistence example
http://scikit-learn.org/stable/modules/model_persistence.html#persistence-example`_
from scikit-learn.