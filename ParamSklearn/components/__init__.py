"""ParamSklearn can be easily extended with new classification and
preprocessing methods. At import time, ParamSklearn checks the directory
``ParamSklearn/components/classification`` for classification algorithms and
``ParamSklearn/components/preprocessing`` for preprocessing algorithms. To be
found, the algorithm must be provide a class implementing one of the given
interfaces.

Coding Guidelines
=================
Please try to adhere to the `scikit-learn coding guidelines <http://scikit-learn.org/stable/developers/index.html#contributing>`_.

Own Implementation of Algorithms
================================
When adding new algorithms, it is possible to implement it directly in the
fit/predict/transform method of a component. We do not recommend this,
but rather recommend to implement an algorithm in a scikit-learn compatible
way (`see here <http://scikit-learn.org/stable/developers/index.html#apis-of-scikit-learn-objects>`_).
Such an implementation should then be put into the `implementation` directory.
and can then be easily wrapped with to become a component in ParamSklearn.

Classification
==============

The ParamSklearnClassificationAlgorithm provides an interface for
Classification Algorithms inside ParamSklearn. It provides four important
functions. Two of them,
:meth:`get_hyperparameter_search_space() <ParamSklearn.components.classification_base.ParamSklearnClassificationAlgorithm.get_hyperparameter_search_space>`
and
:meth:`get_properties() <ParamSklearn.components.classification_base.ParamSklearnClassificationAlgorithm.get_properties>`
are used to
automatically create a valid configuration space. The other two,
:meth:`fit() <ParamSklearn.components.classification_base.ParamSklearnClassificationAlgorithm.fit>` and
:meth:`predict() <ParamSklearn.components.classification_base.ParamSklearnClassificationAlgorithm.predict>`
are an implementation of the `scikit-learn predictor API <http://scikit-learn.org/stable/developers/index.html#apis-of-scikit-learn-objects>`_.

Preprocessing
============="""

from . import classification as classification_components
from . import regression as regression_components
from . import feature_preprocessing as feature_preprocessing_components
from . import data_preprocessing as data_preprocessing_components



