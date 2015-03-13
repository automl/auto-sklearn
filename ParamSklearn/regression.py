from collections import OrderedDict
from itertools import product

import sklearn
if sklearn.__version__ != "0.15.2":
    raise ValueError("ParamSklearn supports only sklearn version 0.15.2, "
                     "you installed %s." % sklearn.__version__)
from sklearn.base import RegressorMixin
import numpy as np

from HPOlibConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction

from ParamSklearn import components as components
from ParamSklearn.base import ParamSklearnBaseEstimator
from ParamSklearn.util import SPARSE
import ParamSklearn.create_searchspace_util


class ParamSklearnRegressor(RegressorMixin, ParamSklearnBaseEstimator):
    """This class implements the regression task.

    It implements a pipeline, which includes one preprocessing step and one
    regression algorithm. It can render a search space including all known
    regression and preprocessing algorithms.

    Contrary to the sklearn API it is not possible to enumerate the
    possible parameters in the __init__ function because we only know the
    available regressors at runtime. For this reason the user must
    specifiy the parameters by passing an instance of
    HPOlibConfigSpace.configuration_space.Configuration.

    Parameters
    ----------
    configuration : HPOlibConfigSpace.configuration_space.Configuration
        The configuration to evaluate.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance
        used by `np.random`.

    Attributes
    ----------
    _estimator : The underlying scikit-learn regression model. This
        variable is assigned after a call to the
        :meth:`ParamSklearn.regression.ParamSklearnRegressor.fit` method.

    _preprocessor : The underlying scikit-learn preprocessing algorithm. This
        variable is only assigned if a preprocessor is specified and
        after a call to the
        :meth:`ParamSklearn.regression.ParamSklearnRegressor.fit` method.

    See also
    --------

    References
    ----------

    Examples
    --------

    """

    def _validate_input_X(self, X):
        # TODO: think of all possible states which can occur and how to
        # handle them
        pass

    def _validate_input_Y(self, Y):
        pass

    def add_model_class(self, model):
        """
        Raises
        ------
            NotImplementedError
        """
        raise NotImplementedError()

    @classmethod
    def get_available_components(cls, available_comp, data_prop, inc, exc):
        components_dict = OrderedDict()
        for name in available_comp:
            if inc is not None and name not in inc:
                continue
            elif exc is not None and name in exc:
                continue
            entry = available_comp[name]

            if not entry.get_properties()['handles_regression']:
                continue
            components_dict[name] = entry
        return components_dict


    @classmethod
    def get_hyperparameter_search_space(cls, include_estimators=None,
                                        exclude_estimators=None,
                                        include_preprocessors=None,
                                        exclude_preprocessors=None,
                                        dataset_properties=None):
        """Return the configuration space for the CASH problem.

        Parameters
        ----------
        include_estimators : list of str
            If include_estimators is given, only the regressors specified
            are used. Specify them by their module name; e.g., to include
            only the SVM use :python:`include_regressors=['svr']`.
            Cannot be used together with :python:`exclude_regressors`.

        exclude_estimators : list of str
            If exclude_estimators is given, only the regressors specified
            are used. Specify them by their module name; e.g., to include
            all regressors except the SVM use
            :python:`exclude_regressors=['svr']`.
            Cannot be used together with :python:`include_regressors`.

        include_preprocessors : list of str
            If include_preprocessors is given, only the preprocessors specified
            are used. Specify them by their module name; e.g., to include
            only the PCA use :python:`include_preprocessors=['pca']`.
            Cannot be used together with :python:`exclude_preprocessors`.

        exclude_preprocessors : list of str
            If include_preprocessors is given, only the preprocessors specified
            are used. Specify them by their module name; e.g., to include
            all preprocessors except the PCA use
            :python:`exclude_preprocessors=['pca']`.
            Cannot be used together with :python:`include_preprocessors`.

        Returns
        -------
        cs : HPOlibConfigSpace.configuration_space.Configuration
            The configuration space describing the ParamSklearnClassifier.
        """
        if include_estimators is not None and exclude_estimators is not None:
            raise ValueError("The arguments include_estimators and "
                             "exclude_regressors cannot be used together.")

        if include_preprocessors is not None and exclude_preprocessors is not None:
            raise ValueError("The arguments include_preprocessors and "
                             "exclude_preprocessors cannot be used together.")

        if dataset_properties is None or not isinstance(dataset_properties, dict):
            dataset_properties = dict()

        if 'sparse' not in dataset_properties:
            # This dataset is probaby dense
            dataset_properties['sparse'] = False

        available_preprocessors = components.preprocessing_components._preprocessors
        preprocessors = ParamSklearnRegressor.get_available_components(
            available_comp=available_preprocessors,
            data_prop=dataset_properties, inc=include_preprocessors,
            exc=exclude_preprocessors)

        # Compile a list of all estimator objects for this problem
        available_regressors = ParamSklearnRegressor._get_estimator_components()
        regressors = ParamSklearnRegressor.get_available_components(
            available_comp=available_regressors, data_prop=dataset_properties,
            inc=include_estimators, exc=exclude_estimators)

        if len(regressors) == 0:
            raise ValueError("No regressors found")
        if len(preprocessors) == 0:
            raise ValueError("No preprocessors found, please add NoPreprocessing")

        preprocessors_list = preprocessors.keys()
        regressors_list = regressors.keys()
        matches = ParamSklearn.create_searchspace_util.get_match_array(
            preprocessors=preprocessors, estimators=regressors,
            sparse=dataset_properties.get('sparse'), pipeline=cls._get_pipeline())

        # Now we have only legal preprocessors/classifiers we combine them
        # Simple sanity checks
        assert np.sum(matches) != 0, "No valid preprocessor/regressor " \
                                     "combination found, probably a bug"
        assert np.sum(matches) <= (matches.shape[0] * matches.shape[1]), \
            "'matches' is not binary; %s <= %d, [%d*%d]" % \
            (str(np.sum(matches)), matches.shape[0]*matches.shape[1],
             matches.shape[0], matches.shape[1])

        if np.sum(matches) < (matches.shape[0] * matches.shape[1]):
            matches, preprocessors_list, regressors_list, preprocessors, regressors = \
                ParamSklearn.create_searchspace_util.sanitize_arrays(
                    m=matches, preprocessors_list=preprocessors_list,
                    estimators_list=regressors_list,
                    preprocessors=preprocessors, estimators=regressors)

        # Sanity checks
        assert len(preprocessors_list) > 0, "No valid preprocessors found"
        assert len(regressors_list) > 0, "No valid classifiers found"

        assert len(preprocessors_list) == matches.shape[0], \
            "Preprocessor deleting went wrong"
        assert len(regressors_list) == matches.shape[1], \
            "Classifier deleting went wrong"
        assert [r in regressors_list for r in regressors]
        assert [p in preprocessors_list for p in preprocessors]

        # Select the default preprocessor before the always active
        # preprocessors are added, so they will not be selected as default
        # preprocessors
        if "no_preprocessing" in preprocessors:
            preprocessor_default = "no_preprocessing"
        else:
            preprocessor_default = sorted(preprocessors.keys())[0]

        # Now add always present preprocessors
        for name in available_preprocessors:
            if name in cls._get_pipeline():
                preprocessors[name] = available_preprocessors[name]

        # Hardcode the defaults based on some educated guesses
        regressor_defaults = ['random_forest', 'liblinear', 'sgd',
                               'libsvm_svc']
        regressor_default = None
        for rd_ in regressor_defaults:
            if rd_ not in regressors:
                continue
            no_preprocessing_idx = preprocessors_list.index(preprocessor_default)
            rd_index = regressors_list.index(rd_)
            if matches[no_preprocessing_idx, rd_index] == 1:
                regressor_default = rd_
                break
        if regressor_default is None:
            regressor_default = regressors.keys()[0]

        # Get the configuration space
        configuration_space = super(ParamSklearnRegressor, cls).\
            get_hyperparameter_search_space(estimator_name=cls._get_estimator_hyperparameter_name(),
                                            default_estimator=regressor_default,
                                            estimator_components=regressors,
                                            default_preprocessor=preprocessor_default,
                                            preprocessor_components=preprocessors,
                                            dataset_properties=dataset_properties,
                                            always_active=cls._get_pipeline())

        # And now add forbidden parameter configurations
        # According to matches
        configuration_space = ParamSklearn.create_searchspace_util.add_forbidden(
            conf_space=configuration_space, preproc_list=preprocessors_list,
            est_list=regressors_list, matches=matches, est_type="regressor")

        # A regressor which can handle sparse data after the densifier
        for key in regressors:
            if SPARSE in regressors[key].get_properties()['input']:
                try:
                    configuration_space.add_forbidden_clause(
                        ForbiddenAndConjunction(
                            ForbiddenEqualsClause(
                                configuration_space.get_hyperparameter(
                                    'regressor'), key),
                            ForbiddenEqualsClause(
                                configuration_space.get_hyperparameter(
                                    'preprocessor'), 'densifier')
                        ))
                except:
                    pass


        # which would take too long
        # Combinations of tree-based models with feature learning:
        regressors_ = ["random_forest", "gradient_boosting", "gaussian_process"]
        feature_learning_ = ["kitchen_sinks", "sparse_filtering"]

        for r, f in product(regressors_, feature_learning_):
            if r not in regressors_list:
                continue
            if f not in preprocessors_list:
                continue
            try:
                configuration_space.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "regressor"), r),
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "preprocessor"), f)))
            except:
                pass

        return configuration_space

    @staticmethod
    def _get_estimator_components():
        return components.regression_components._regressors

    @staticmethod
    def _get_estimator_hyperparameter_name():
        return "regressor"

    @staticmethod
    def _get_pipeline():
        return ["imputation", "rescaling", "__preprocessor__", "__estimator__"]
