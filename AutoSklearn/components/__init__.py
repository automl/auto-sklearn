"""AutoSklearn can be easily extended with new classification and
preprocessing methods. At import time, AutoSklearn checks the directory
``AutoSklearn/components/classification`` for classification algorithms and
``AutoSklearn/components/preprocessing`` for preprocessing algorithms. To be
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
and can then be easily wrapped with to become a component in AutoSklearn.

Classification
==============

The AutoSklearnClassificationAlgorithm provides an interface for
Classification Algorithms inside AutoSklearn. It provides four important
functions. Two of them,
:meth:`get_hyperparameter_search_space() <AutoSklearn.components.classification_base.AutoSklearnClassificationAlgorithm.get_hyperparameter_search_space>`
and
:meth:`get_properties() <AutoSklearn.components.classification_base.AutoSklearnClassificationAlgorithm.get_properties>`
are used to
automatically create a valid configuration space. The other two,
:meth:`fit() <AutoSklearn.components.classification_base.AutoSklearnClassificationAlgorithm.fit>` and
:meth:`predict() <AutoSklearn.components.classification_base.AutoSklearnClassificationAlgorithm.predict>`
are an implementation of the `scikit-learn predictor API <http://scikit-learn.org/stable/developers/index.html#apis-of-scikit-learn-objects>`_.

Preprocessing
============="""

from . import classification as classification_components
from . import regression as regression_components
from . import preprocessing as preprocessing_components
