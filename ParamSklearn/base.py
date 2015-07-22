from abc import ABCMeta, abstractmethod
from collections import defaultdict, OrderedDict
import copy

import numpy as np
import sklearn
if sklearn.__version__ != "0.16.1":
    raise ValueError("ParamSklearn supports only sklearn version 0.16.1, "
                     "you installed %s." % sklearn.__version__)

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_random_state, check_is_fitted

from ParamSklearn import components as components
import ParamSklearn.create_searchspace_util


class ParamSklearnBaseEstimator(BaseEstimator):
    """Base class for all ParamSklearn task models.

    Notes
    -----
    This class should not be instantiated, only subclassed."""
    __metaclass__ = ABCMeta

    def __init__(self, configuration, random_state=None):
        self.configuration = configuration

        if random_state is None:
            self.random_state = check_random_state(1)
        else:
            self.random_state = check_random_state(random_state)

    def fit(self, X, y, fit_params=None, init_params=None):
        """Fit the selected algorithm to the training data.

        Parameters
        ----------
        X : array-like or sparse, shape = (n_samples, n_features)
            Training data. The preferred type of the matrix (dense or sparse)
            depends on the estimator selected.

        y : array-like
            Targets

        fit_params : dict
            See the documentation of sklearn.pipeline.Pipeline for formatting
            instructions.

        init_params : dict
            Pass arguments to the constructors of single methods. To pass
            arguments to only one of the methods (lets says the
            OneHotEncoder), seperate the class name from the argument by a ':'.

        Returns
        -------
        self : returns an instance of self.

        Raises
        ------
        NoModelException
            NoModelException is raised if fit() is called without specifying
            a classification algorithm first.
        """
        X, fit_params = self.pre_transform(X, y, fit_params=fit_params,
                                          init_params=init_params)
        self.fit_estimator(X, y, fit_params=fit_params)
        return self

    def pre_transform(self, X, y, fit_params=None, init_params=None):

        # Save all transformation object in a list to create a pipeline object
        steps = []

        # seperate the init parameters for the single methods
        init_params_per_method = defaultdict(dict)
        if init_params is not None and len(init_params) != 0:
            for init_param, value in init_params.items():
                method, param = init_param.split(":")
                init_params_per_method[method][param] = value

        # List of preprocessing steps (and their order)
        preprocessors_names = [preprocessor[0] for
                               preprocessor in self._get_pipeline()[:-1]]

        for preproc_name in preprocessors_names:
            preproc_params = {}

            for instantiated_hyperparameter in self.configuration:
                if not instantiated_hyperparameter.startswith(
                        preproc_name + ":"):
                    continue
                if self.configuration[instantiated_hyperparameter] is None:
                    continue

                name_ = instantiated_hyperparameter.split(":")[-1]
                preproc_params[name_] = self.configuration[
                    instantiated_hyperparameter]

            preproc_params.update(init_params_per_method[preproc_name])

            preprocessor_object = components.preprocessing_components. \
                _preprocessors[preproc_name](random_state=self.random_state,
                                             **preproc_params)

            # Ducktyping...
            if hasattr(preprocessor_object, 'get_components'):
                preprocessor_object = preprocessor_object.choice

            steps.append((preproc_name, preprocessor_object))

        # Extract Estimator Hyperparameters from the configuration object
        estimator_name = self._get_pipeline()[-1][0]
        estimator_object = self._get_pipeline()[-1][1]
        estimator_parameters = {}
        for instantiated_hyperparameter in self.configuration:
            if not instantiated_hyperparameter.startswith(estimator_name):
                continue
            if self.configuration[instantiated_hyperparameter] is None:
                continue

            name_ = instantiated_hyperparameter.split(":")[-1]
            estimator_parameters[name_] = self.configuration[
                instantiated_hyperparameter]

        estimator_parameters.update(init_params_per_method[estimator_name])
        estimator_object = estimator_object(random_state=self.random_state,
                            **estimator_parameters)

        # Ducktyping...
        if hasattr(estimator_object, 'get_components'):
            estimator_object = estimator_object.choice

        steps.append((estimator_name, estimator_object))

        self.pipeline_ = Pipeline(steps)
        if fit_params is None or not isinstance(fit_params, dict):
            fit_params = dict()
        else:
            fit_params = {key.replace(":", "__"): value for key, value in
                          fit_params.items()}
        X, fit_params = self.pipeline_._pre_transform(X, y, **fit_params)
        return X, fit_params

    def fit_estimator(self, X, y, fit_params=None):
        check_is_fitted(self, 'pipeline_')
        if fit_params is None:
            fit_params = {}
        self.pipeline_.steps[-1][-1].fit(X, y, **fit_params)
        return self

    def iterative_fit(self, X, y, fit_params=None, n_iter=1):
        check_is_fitted(self, 'pipeline_')
        if fit_params is None:
            fit_params = {}
        self.pipeline_.steps[-1][-1].iterative_fit(X, y, n_iter=n_iter,
                                                   **fit_params)

    def estimator_supports_iterative_fit(self):
        check_is_fitted(self, 'pipeline_')
        return hasattr(self.pipeline_.steps[-1][-1], 'iterative_fit')

    def configuration_fully_fitted(self):
        check_is_fitted(self, 'pipeline_')
        return self.pipeline_.steps[-1][-1].configuration_fully_fitted()

    def predict(self, X, batch_size=None):
        """Predict the classes using the selected model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        batch_size: int or None, defaults to None
            batch_size controls whether the ParamSklearn pipeline will be
            called on small chunks of the data. Useful when calling the
            predict method on the whole array X results in a MemoryError.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Returns the predicted values"""
        # TODO check if fit() was called before...

        if batch_size is None:
            return self.pipeline_.predict(X)
        else:
            if type(batch_size) is not int or batch_size <= 0:
                raise Exception("batch_size must be a positive integer")

            else:
                if self.num_targets == 1:
                    y = np.zeros((X.shape[0],))
                else:
                    y = np.zeros((X.shape[0], self.num_targets))

                # Copied and adapted from the scikit-learn GP code
                for k in range(max(1, int(np.ceil(float(X.shape[0]) /
                                                  batch_size)))):
                    batch_from = k * batch_size
                    batch_to = min([(k + 1) * batch_size, X.shape[0]])
                    y[batch_from:batch_to] = \
                        self.predict(X[batch_from:batch_to], batch_size=None)

                return y

    @classmethod
    def get_hyperparameter_search_space(cls, include=None, exclude=None,
                                        dataset_properties=None):
        """Return the configuration space for the CASH problem.

        This method should be called by the method
        get_hyperparameter_search_space of a subclass. After the subclass
        assembles a list of available estimators and preprocessor components,
        _get_hyperparameter_search_space can be called to do the work of
        creating the actual
        HPOlibConfigSpace.configuration_space.ConfigurationSpace object.

        Parameters
        ----------
        estimator_name : str
            Name of the estimator hyperparameter which will be used in the
            configuration space. For a classification task, this would be
            'classifier'.

        estimator_components : dict {name: component}
            Dictionary with all estimator components to be included in the
            configuration space.

        preprocessor_components : dict {name: component}
            Dictionary with all preprocessor components to be included in the
            configuration space. .

        always_active : list of str
            A list of components which will always be active in the pipeline.
            This is useful for components like imputation which have
            hyperparameters to be configured, but which do not have any parent.

        default_estimator : str
            Default value for the estimator hyperparameter.

        Returns
        -------
        cs : HPOlibConfigSpace.configuration_space.Configuration
            The configuration space describing the ParamSklearnClassifier.

        """
        raise NotImplementedError()

    @classmethod
    def _get_hyperparameter_search_space(cls, cs, dataset_properties, exclude,
                                         include, pipeline):
        for node_0_idx, node_1_idx in zip(range(len(pipeline) - 1),
                                          range(1, len(pipeline))):
            node_0_name = pipeline[node_0_idx][0]
            node_1_name = pipeline[node_1_idx][0]
            node_0 = pipeline[node_0_idx][1]
            node_1 = pipeline[node_1_idx][1]

            node_0_include = include.get(
                node_0_name) if include is not None else None
            node_0_exclude = exclude.get(
                node_0_name) if exclude is not None else None
            node_1_include = include.get(
                node_1_name) if include is not None else None
            node_1_exclude = exclude.get(
                node_1_name) if exclude is not None else None

            matches = ParamSklearn.create_searchspace_util.get_match_array(
                node_0=node_0, node_1=node_1, node_0_include=node_0_include,
                node_0_exclude=node_0_exclude, node_1_include=node_1_include,
                node_1_exclude=node_1_exclude,
                dataset_properties=dataset_properties, )

            # Now we have only legal combinations at this step of the pipeline
            # Simple sanity checks
            assert np.sum(matches) != 0, "No valid %s/%s combination found, " \
                                         "probably a bug." % (node_0_name,
                                                              node_1_name)

            assert np.sum(matches) <= (matches.shape[0] * matches.shape[1]), \
                "'matches' is not binary; %s <= %d, [%d*%d]" % \
                (str(np.sum(matches)), matches.shape[0] * matches.shape[1],
                 matches.shape[0], matches.shape[1])

            if np.sum(matches) < (matches.shape[0] * matches.shape[1]):
                matches, node_0_list, node_1_list = \
                    ParamSklearn.create_searchspace_util.sanitize_arrays(
                        matches=matches, node_0=node_0, node_1=node_1,
                        dataset_properties=dataset_properties,
                        node_0_include=node_0_include,
                        node_0_exclude=node_0_exclude,
                        node_1_include=node_1_include,
                        node_1_exclude=node_1_exclude)

                # Check if we reached a dead end
                assert len(node_0_list) > 0, "No valid node 0 found"
                assert len(node_1_list) > 0, "No valid node 1 found"

                # Check for inconsistencies
                assert len(node_0_list) == matches.shape[0], \
                    "Node 0 deleting went wrong"
                assert len(node_1_list) == matches.shape[1], \
                    "Node 1 deleting went wrong"
            else:
                if hasattr(node_0, "get_components"):
                    node_0_list = node_0.get_available_components(
                        data_prop=dataset_properties,
                        include=node_0_include,
                        exclude=node_0_exclude
                    )
                else:
                    node_0_list = None
                if hasattr(node_1, "get_components"):
                    node_1_list = node_1.get_available_components(
                        data_prop=dataset_properties,
                        include=node_1_include,
                        exclude=node_1_exclude
                    )
                else:
                    node_1_list = None

            if hasattr(node_0, "get_components"):
                node_0_name += ":__choice__"

            if node_0_idx == 0:
                if hasattr(node_0, "get_components"):
                    cs.add_configuration_space(node_0_name,
                                               node_0.get_hyperparameter_search_space(
                                                   dataset_properties,
                                                   include=node_0_list))
                else:
                    cs.add_configuration_space(node_0_name,
                                               node_0.get_hyperparameter_search_space(
                                                   dataset_properties))

            if hasattr(node_1, "get_components"):
                cs.add_configuration_space(node_1_name,
                                           node_1.get_hyperparameter_search_space(
                                               dataset_properties,
                                               include=node_1_list))
                node_1_name += ":__choice__"
            else:
                cs.add_configuration_space(node_1_name,
                                           node_1.get_hyperparameter_search_space(
                                               dataset_properties))

            # And now add forbidden parameter configurations
            # According to matches
            if np.sum(matches) < (matches.shape[0] * matches.shape[1]):
                cs = ParamSklearn.create_searchspace_util.add_forbidden(
                    conf_space=cs, node_0_list=node_0_list,
                    node_1_list=node_1_list, matches=matches,
                    node_0_name=node_0_name, node_1_name=node_1_name)
        return cs

    @staticmethod
    def _get_pipeline():
        raise NotImplementedError()

    def _get_estimator_hyperparameter_name(self):
        raise NotImplementedError()


