from abc import ABCMeta
from collections import defaultdict

import numpy as np
from ConfigSpace import Configuration
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_random_state, check_is_fitted

from .components.base import AutoSklearnChoice, AutoSklearnComponent
import autosklearn.pipeline.create_searchspace_util


class BasePipeline(Pipeline):
    """Base class for all pipeline objects.

    Notes
    -----
    This class should not be instantiated, only subclassed."""
    __metaclass__ = ABCMeta

    def __init__(self, config=None, pipeline=None, dataset_properties=None,
                 include=None, exclude=None, random_state=None,
                 init_params=None):

        self._init_params = init_params if init_params is not None else {}
        self.include_ = include if include is not None else {}
        self.exclude_ = exclude if exclude is not None else {}
        self.dataset_properties_ = dataset_properties if \
            dataset_properties is not None else {}

        if pipeline is None:
            self.steps = self._get_pipeline()
        else:
            self.steps = pipeline

        self.config_space = self.get_hyperparameter_search_space()

        if config is None:
            self.configuration_ = self.config_space.get_default_configuration()
        else:
            if isinstance(config, dict):
                config = Configuration(self.config_space, config)
            if self.config_space != config.configuration_space:
                print(self.config_space._children)
                print(config.configuration_space._children)
                import difflib
                diff = difflib.unified_diff(
                    str(self.config_space).splitlines(),
                    str(config.configuration_space).splitlines())
                diff = '\n'.join(diff)
                raise ValueError('Configuration passed does not come from the '
                                 'same configuration space. Differences are: '
                                 '%s' % diff)
            self.configuration_ = config

        self.set_hyperparameters(self.configuration_, init_params=init_params)

        if random_state is None:
            self.random_state = check_random_state(1)
        else:
            self.random_state = check_random_state(random_state)
        super().__init__(steps=self.steps)

        self._additional_run_info = {}

    def fit(self, X, y, **fit_params):
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

        Returns
        -------
        self : returns an instance of self.

        Raises
        ------
        NoModelException
            NoModelException is raised if fit() is called without specifying
            a classification algorithm first.
        """
        X, fit_params = self.fit_transformer(X, y, **fit_params)
        self.fit_estimator(X, y, **fit_params)
        return self

    def fit_transformer(self, X, y, fit_params=None):
        self.num_targets = 1 if len(y.shape) == 1 else y.shape[1]
        if fit_params is None:
            fit_params = {}
        fit_params = {key.replace(":", "__"): value for key, value in
                      fit_params.items()}
        Xt, fit_params = self._fit(X, y, **fit_params)
        if fit_params is None:
            fit_params = {}
        return Xt, fit_params

    def fit_estimator(self, X, y, **fit_params):
        fit_params = {key.replace(":", "__"): value for key, value in
                      fit_params.items()}
        self._final_estimator.fit(X, y, **fit_params)
        return self

    def iterative_fit(self, X, y, n_iter=1, **fit_params):
        self._final_estimator.iterative_fit(X, y, n_iter=n_iter,
                                            **fit_params)

    def estimator_supports_iterative_fit(self):
        return self._final_estimator.estimator_supports_iterative_fit()

    def configuration_fully_fitted(self):
        return self._final_estimator.configuration_fully_fitted()

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

        if batch_size is None:
            return super().predict(X).astype(self._output_dtype)
        else:
            if not isinstance(batch_size, int):
                raise ValueError("Argument 'batch_size' must be of type int, "
                                 "but is '%s'" % type(batch_size))
            if batch_size <= 0:
                raise ValueError("Argument 'batch_size' must be positive, "
                                 "but is %d" % batch_size)

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

    def set_hyperparameters(self, configuration, init_params=None):
        self.configuration = configuration

        for node_idx, n_ in enumerate(self.steps):
            node_name, node = n_

            sub_configuration_space = node.get_hyperparameter_search_space(
                dataset_properties=self.dataset_properties_
            )
            sub_config_dict = {}
            for param in configuration:
                if param.startswith('%s:' % node_name):
                    value = configuration[param]
                    new_name = param.replace('%s:' % node_name, '', 1)
                    sub_config_dict[new_name] = value

            sub_configuration = Configuration(sub_configuration_space,
                                              values=sub_config_dict)

            if init_params is not None:
                sub_init_params_dict = {}
                for param in init_params:
                    if param.startswith('%s:' % node_name):
                        value = init_params[param]
                        new_name = param.replace('%s:' % node_name, '', 1)
                        sub_init_params_dict[new_name] = value
            else:
                sub_init_params_dict = None

            if isinstance(node, (AutoSklearnChoice, AutoSklearnComponent)):
                node.set_hyperparameters(configuration=sub_configuration,
                                         init_params=sub_init_params_dict)
            else:
                raise NotImplementedError('Not supported yet!')

        return self

    def get_hyperparameter_search_space(self):
        """Return the configuration space for the CASH problem.

        Returns
        -------
        cs : ConfigSpace.configuration_space.Configuration
            The configuration space describing the AutoSklearnClassifier.

        """
        if not hasattr(self, 'config_space') or self.config_space is None:
            self.config_space = self._get_hyperparameter_search_space(
                include=self.include_, exclude=self.exclude_,
                dataset_properties=self.dataset_properties_)
        return self.config_space

    def _get_hyperparameter_search_space(self, include=None, exclude=None,
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

    def _get_base_search_space(self, cs, dataset_properties, exclude,
                               include, pipeline):
        if include is None:
            if self.include_ is None:
                include = {}
            else:
                include = self.include_

        keys = [pair[0] for pair in pipeline]
        for key in include:
            if key not in keys:
                raise ValueError('Invalid key in include: %s; should be one '
                                 'of %s' % (key, keys))

        if exclude is None:
            if self.exclude_ is None:
                exclude = {}
            else:
                exclude = self.exclude_

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

            is_choice = isinstance(node, AutoSklearnChoice)

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
                sub_config_space = node.get_hyperparameter_search_space(
                    dataset_properties, include=choices_list)
                cs.add_configuration_space(node_name, sub_config_space)

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
        self.configuration_._populate_values()
        for hp_name in self.configuration_:
            if self.configuration_[hp_name] is not None:
                configuration[hp_name] = self.configuration_[hp_name]

        configuration_string = ''.join(
            ['configuration={\n  ',
             ',\n  '.join(["'%s': %s" % (hp_name, repr(configuration[hp_name]))
                                         for hp_name in sorted(configuration)]),
             '}'])

        if len(self.dataset_properties_) > 0:
            dataset_properties_string = []
            dataset_properties_string.append('dataset_properties={')
            for i, item in enumerate(self.dataset_properties_.items()):
                if i != 0:
                    dataset_properties_string.append(',\n  ')
                else:
                    dataset_properties_string.append('\n  ')

                if isinstance(item[1], str):
                    dataset_properties_string.append("'%s': '%s'" % (item[0],
                                                                     item[1]))
                else:
                    dataset_properties_string.append("'%s': %s" % (item[0],
                                                                   item[1]))
            dataset_properties_string.append('}')
            dataset_properties_string = ''.join(dataset_properties_string)

            rval = '%s(%s,\n%s)' % (class_name, configuration,
                                    dataset_properties_string)
        else:
            rval = '%s(%s)' % (class_name, configuration_string)

        return rval

    def _get_pipeline(self):
        raise NotImplementedError()

    def _get_estimator_hyperparameter_name(self):
        raise NotImplementedError()

    def get_additional_run_info(self):
        """Allows retrieving additional run information from the pipeline.

        Can be overridden by subclasses to return additional information to
        the optimization algorithm.
        """
        return self._additional_run_info

