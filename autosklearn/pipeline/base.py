from abc import ABCMeta
from collections import defaultdict

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_random_state, check_is_fitted

from autosklearn.pipeline import components as components
import autosklearn.pipeline.create_searchspace_util


class BasePipeline(BaseEstimator):
    """Base class for all pipeline objects.

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
        self.fit_estimator(X, y, **fit_params)
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

        # Instantiate preprocessor objects
        for preproc_name, preproc_class in self._get_pipeline()[:-1]:
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

            preprocessor_object = preproc_class(
                random_state=self.random_state, **preproc_params)

            # Ducktyping...
            if hasattr(preproc_class, 'get_components'):
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

    def fit_estimator(self, X, y, **fit_params):
        check_is_fitted(self, 'pipeline_')
        if fit_params is None:
            fit_params = {}
        self.pipeline_.steps[-1][-1].fit(X, y, **fit_params)
        return self

    def iterative_fit(self, X, y, n_iter=1, **fit_params):
        check_is_fitted(self, 'pipeline_')
        if fit_params is None:
            fit_params = {}
        self.pipeline_.steps[-1][-1].iterative_fit(X, y, n_iter=n_iter,
                                                   **fit_params)

    def estimator_supports_iterative_fit(self):
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
            batch_size controls whether the pipeline will be
            called on small chunks of the data. Useful when calling the
            predict method on the whole array X results in a MemoryError.

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Returns the predicted values"""
        # TODO check if fit() was called before...

        if batch_size is None:
            return self.pipeline_.predict(X).astype(self._output_dtype)
        else:
            if type(batch_size) is not int or batch_size <= 0:
                raise Exception("batch_size must be a positive integer")

            else:
                if self.num_targets == 1:
                    y = np.zeros((X.shape[0],), dtype=self._output_dtype)
                else:
                    y = np.zeros((X.shape[0], self.num_targets),
                                 dtype=self._output_dtype)

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
        ConfigSpace.configuration_space.ConfigurationSpace object.

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
        cs : ConfigSpace.configuration_space.Configuration
            The configuration space describing the AutoSklearnClassifier.

        """
        raise NotImplementedError()

    @classmethod
    def _get_hyperparameter_search_space(cls, cs, dataset_properties, exclude,
                                         include, pipeline):
        if include is None:
            include = {}

        keys = [pair[0] for pair in pipeline]
        for key in include:
            if key not in keys:
                raise ValueError('Invalid key in include: %s; should be one '
                                 'of %s' % (key, keys))

        if exclude is None:
            exclude = {}

        keys = [pair[0] for pair in pipeline]
        for key in exclude:
            if key not in keys:
                raise ValueError('Invalid key in exclude: %s; should be one '
                                 'of %s' % (key, keys))

        if 'sparse' not in dataset_properties:
            # This dataset is probaby dense
            dataset_properties['sparse'] = False
        if 'signed' not in dataset_properties:
            # This dataset probably contains unsigned data
            dataset_properties['signed'] = False

        matches = autosklearn.pipeline.create_searchspace_util.get_match_array(
            pipeline, dataset_properties, include=include, exclude=exclude)

        # Now we have only legal combinations at this step of the pipeline
        # Simple sanity checks
        assert np.sum(matches) != 0, "No valid pipeline found."

        assert np.sum(matches) <= np.size(matches), \
            "'matches' is not binary; %s <= %d, %s" % \
            (str(np.sum(matches)), np.size(matches), str(matches.shape))

        # Iterate each dimension of the matches array (each step of the
        # pipeline) to see if we can add a hyperparameter for that step
        for node_idx, n_ in enumerate(pipeline):
            node_name, node = n_
            is_choice = hasattr(node, "get_available_components")

            # if the node isn't a choice we can add it immediately because it
            #  must be active (if it wouldn't, np.sum(matches) would be zero
            if not is_choice:
                cs.add_configuration_space(node_name,
                    node.get_hyperparameter_search_space(dataset_properties))
            # If the node isn't a choice, we have to figure out which of it's
            #  choices are actually legal choices
            else:
                choices_list = autosklearn.pipeline.create_searchspace_util.\
                    find_active_choices(matches, node, node_idx,
                                        dataset_properties,
                                        include.get(node_name),
                                        exclude.get(node_name))
                cs.add_configuration_space(node_name,
                    node.get_hyperparameter_search_space(
                        dataset_properties, include=choices_list))
        # And now add forbidden parameter configurations
        # According to matches
        if np.sum(matches) < np.size(matches):
            cs = autosklearn.pipeline.create_searchspace_util.add_forbidden(
                conf_space=cs, pipeline=pipeline, matches=matches,
                dataset_properties=dataset_properties, include=include,
                exclude=exclude)

        return cs

    def __repr__(self):
        class_name = self.__class__.__name__

        configuration = {}
        self.configuration._populate_values()
        for hp_name in self.configuration:
            if self.configuration[hp_name] is not None:
                configuration[hp_name] = self.configuration[hp_name]

        configuration_string = ''.join(
            ['configuration={\n  ',
             ',\n  '.join(["'%s': %s" % (hp_name, repr(configuration[hp_name]))
                                         for hp_name in sorted(configuration)]),
             '}'])

        return '%s(%s)' % (class_name, configuration_string)

    @classmethod
    def _get_pipeline(cls):
        if cls == autosklearn.pipelineBaseEstimator:
            return []
        raise NotImplementedError()

    def _get_estimator_hyperparameter_name(self):
        raise NotImplementedError()

