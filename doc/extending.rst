:orphan:

.. _extending:

======================
Extending auto-sklearn
======================

auto-sklearn can be easily extended with new classification, regression and
feature preprocessing methods. In order to do so, a user has to implement a
wrapper class and register it to auto-sklearn. This manual will walk you
through the process.


Writing a component
===================

Depending on the purpose, the component has to be a subclass of one of the
following base classes:

* classification: :class:`autosklearn.pipeline.components.base.AutoSklearnClassificationAlgorithm`
* regression: :class:`autosklearn.pipeline.components.base.AutoSklearnRegressionAlgorithm`
* preprocessing: :class:`autosklearn.pipeline.components.base.AutoSklearnPreprocessingAlgorithm`

In general, these classes are wrappers around existing machine learning
models and only add the functionality auto-sklearn needs. Of course you can
also implement a machine learning algorithm directly inside a component.

Each component has to implement a method which returns its configuration
space, a method for querying properties of the component and methods like
`fit()`, `predict()` or `transform()` based on the task of the component.
These are described in the subsections
:ref:`get_hyperparameter_search_space` and :ref:`get_properties`

After writing a component class, you have to tell auto-sklearn about its
existence. You have to add it with the following function calls, depending on
the type of component:

.. autofunction:: autosklearn.pipeline.components.classification.add_classifier

.. autofunction:: autosklearn.pipeline.components.regression.add_regressor

.. autofunction:: autosklearn.pipeline.components.feature_preprocessing.add_preprocessor


.. _get_hyperparameter_search_space:

get_hyperparameter_search_space()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Return an instance of ``HPOlibConfigSpace.configuration_space
.ConfigurationSpace``.

See also the abstract definitions:
:meth:`AutoSklearnClassificationAlgorithm.get_hyperparameter_search_space() <autosklearn.pipeline.components.base.AutoSklearnClassificationAlgorithm.get_hyperparameter_search_space>`
:meth:`AutoSklearnRegressionAlgorithm.get_hyperparameter_search_space() <autosklearn.pipeline.components.base.AutoSklearnRegressionAlgorithm.get_hyperparameter_search_space>`
:meth:`AutoSklearnPreprocessingAlgorithm.get_hyperparameter_search_space() <autosklearn.pipeline.components.base.AutoSklearnPreprocessingAlgorithm.get_hyperparameter_search_space>`

To find out about how to create a ``ConfigurationSpace``-object, please look
at the source code on `github.com <https://github.com/automl/auto-sklearn/tree/master/autosklearn/pipeline/components/classification>`_.

.. _get_properties:

get_properties()
~~~~~~~~~~~~~~~~

Return a dictionary which defines how the component can be used when
constructing a machine learning pipeline. The following fields must be
specified:

* shortname : str
    an abbreviation of the component
* name : str
    the full name of the component
* handles_regression : bool
    whether the component can handle regression data
* handles_classification : bool
    whether the component can handle classification data
* handles_multiclass : bool
    whether the component can handle multiclass classification data
* handles_multilabel : bool
    whether the component can multilabel classification data
* is_deterministic : bool
    whether the component gives the same result when using several times,
    but with the same random seed
* input : tuple
    type of input data the component can handle, can have multiple values:

    * **autosklearn.constants.DENSE**
        dense data arrays, mutually exclusive with autosklearn.constants.SPARSE
    * **autosklearn.constants.SPARSE**
        sparse data matrices, mutually exclusive with autosklearn.constants.DENSE
    * **autosklearn.constants.UNSIGNED_DATA**
        unsigned data array, meaning only positive input, mutually exclusive
        with autosklearn.constants.SIGNED_DATA
    * **autosklearn.constants.SIGNED_DATA**
        signed data array, meaning both positive and negative input values,
        mutually exclusive with autosklearn.constants.UNSIGNED_DATA
* output : tuple
    type of output data the component produces

    * **autosklearn.constants.PREDICTIONS**
        predictions, for example by a classifier
    * **autosklearn.constants.INPUT**
        data in the same form as the input
    * **autosklearn.constants.DENSE**
        dense data arrays, mutually exclusive with autosklearn.constants.SPARSE.
        This implies that sparse data will be converted into a dense
        representation.
    * **autosklearn.constants.SPARSE**
        sparse data matrices, mutually exclusive with
        autosklearn.constants.DENSE. This implies that dense data will
        be converted into a sparse representation
    * **autosklearn.constants.UNSIGNED_DATA**
        unsigned data array, meaning only positive input, mutually exclusive
        with autosklearn.constants.SIGNED_DATA. This allows for algorithms which
        can only work on positive data.
    * **autosklearn.constants.SIGNED_DATA**
        signed data array, meaning both positive and negative input values,
        mutually exclusive with autosklearn.constants.UNSIGNED_DATA

Classification
==============

In addition two `get_properties()` and `get_hyperparameter_search_space()`
you have to implement
:meth:`AutoSklearnClassificationAlgorithm.fit() <autosklearn.pipeline.components.base.AutoSklearnClassificationAlgorithm.fit>`
and
:meth:`AutoSklearnClassificationAlgorithm.predict() <autosklearn.pipeline.components.base.AutoSklearnClassificationAlgorithm.predict>`
. These are an implementation of the `scikit-learn predictor API
<http://scikit-learn.org/stable/developers/index.html#apis-of-scikit-learn-objects>`_.

Regression
==========

In addition two `get_properties()` and `get_hyperparameter_search_space()`
you have to implement
:meth:`AutoSklearnRegressionAlgorithm.fit() <autosklearn.pipeline.components.base.AutoSklearnRegressionAlgorithm.fit>`
and
:meth:`AutoSklearnRegressionAlgorithm.predict() <autosklearn.pipeline.components.base.AutoSklearnRegressionAlgorithm.predict>`
. These are an implementation of the `scikit-learn predictor API
<http://scikit-learn.org/stable/developers/index.html#apis-of-scikit-learn-objects>`_.

Feature Preprocessing
=====================

In addition two `get_properties()` and `get_hyperparameter_search_space()`
you have to implement
:meth:`AutoSklearnPreprocessingAlgorithm.fit() <autosklearn.pipeline.components.base.AutoSklearnPreprocessingAlgorithm.fit>`
and
:meth:`AutoSklearnPreprocessingAlgorithm.transform() <autosklearn.pipeline.components.base.AutoSklearnPreprocessingAlgorithm.transform>`
. These are an implementation of the `scikit-learn predictor API
<http://scikit-learn.org/stable/developers/index.html#apis-of-scikit-learn-objects>`_.
