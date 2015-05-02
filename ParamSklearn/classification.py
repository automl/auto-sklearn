from collections import OrderedDict
from itertools import product

import numpy as np

from sklearn.base import ClassifierMixin

from HPOlibConfigSpace.forbidden import ForbiddenEqualsClause
from HPOlibConfigSpace.forbidden import ForbiddenAndConjunction

from ParamSklearn import components as components
from ParamSklearn.base import ParamSklearnBaseEstimator
from ParamSklearn.util import SPARSE
from ParamSklearn.components.preprocessing.balancing import Balancing
import ParamSklearn.create_searchspace_util


class ParamSklearnClassifier(ClassifierMixin, ParamSklearnBaseEstimator):
    """This class implements the classification task.

    It implements a pipeline, which includes one preprocessing step and one
    classification algorithm. It can render a search space including all known
    classification and preprocessing algorithms.

    Contrary to the sklearn API it is not possible to enumerate the
    possible parameters in the __init__ function because we only know the
    available classifiers at runtime. For this reason the user must
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
    _estimator : The underlying scikit-learn classification model. This
        variable is assigned after a call to the
        :meth:`ParamSklearn.classification.ParamSklearnClassifier.fit` method.

    _preprocessor : The underlying scikit-learn preprocessing algorithm. This
        variable is only assigned if a preprocessor is specified and
        after a call to the
        :meth:`ParamSklearn.classification.ParamSklearnClassifier.fit` method.

    See also
    --------

    References
    ----------

    Examples
    --------

    """

    def fit(self, X, Y, fit_params=None, init_params=None):
        self.num_targets = 1 if len(Y.shape) == 1 else Y.shape[1]

        # Weighting samples has to be done here, not in the components
        if self.configuration['balancing:strategy'].value == 'weighting':
            balancing = Balancing(strategy='weighting')
            init_params, fit_params = balancing.get_weights(
                Y, self.configuration['classifier'].value,
                self.configuration['preprocessor'].value,
                init_params, fit_params)

        super(ParamSklearnClassifier, self).fit(X, Y, fit_params=fit_params,
                                                init_params=init_params)

        return self

    def predict_proba(self, X, batch_size=None):
        """predict_proba.

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
        """
        if batch_size is None:
            self._validate_input_X(X)
            Xt = X
            for name, transform in self._pipeline.steps[:-1]:
                Xt = transform.transform(Xt)

            return self._pipeline.steps[-1][-1].predict_proba(Xt)

        else:
            if type(batch_size) is not int or batch_size <= 0:
                raise Exception("batch_size must be a positive integer")

            else:
                # Probe for the target array dimensions
                target = self.predict_proba(X[0].copy())

                # Binary or Multiclass
                if len(target) == 1:
                    y = np.zeros((X.shape[0], target.shape[1]))

                    for k in range(max(1, int(np.ceil(float(X.shape[0]) /
                            batch_size)))):
                        batch_from = k * batch_size
                        batch_to = min([(k + 1) * batch_size, X.shape[0]])
                        y[batch_from:batch_to] = \
                            self.predict_proba(X[batch_from:batch_to],
                                               batch_size=None)

                elif len(target) > 1:
                    y = [np.zeros((X.shape[0], target[i].shape[1]))
                         for i in range(len(target))]

                    for k in range(max(1, int(np.ceil(float(X.shape[0]) /
                            batch_size)))):
                        batch_from = k * batch_size
                        batch_to = min([(k + 1) * batch_size, X.shape[0]])
                        predictions = \
                            self.predict_proba(X[batch_from:batch_to],
                                               batch_size=None)

                        for i in range(len(target)):
                            y[i][batch_from:batch_to] = predictions[i]

                return y

    @classmethod
    def get_available_components(cls, available_comp, data_prop, inc, exc):
        components_dict = OrderedDict()
        for name in available_comp:
            if inc is not None and name not in inc:
                continue
            elif exc is not None and name in exc:
                continue

            entry = available_comp[name]
            if entry.get_properties()['handles_classification'] is False:
                continue
            if data_prop.get('multiclass') is True and entry.get_properties()[
                    'handles_multiclass'] is False:
                continue
            if data_prop.get('multilabel') is True and available_comp[name]. \
                    get_properties()['handles_multilabel'] is False:
                continue
            components_dict[name] = entry

        return components_dict

    @classmethod
    def get_hyperparameter_search_space(cls, include_estimators=None,
                                        exclude_estimators=None,
                                        include_preprocessors=None,
                                        exclude_preprocessors=None,
                                        dataset_properties=None):

        if include_estimators is not None and exclude_estimators is not None:
            raise ValueError("The arguments include_estimators and "
                             "exclude_estimators cannot be used together.")

        if include_preprocessors is not None and exclude_preprocessors is not None:
            raise ValueError("The arguments include_preprocessors and "
                             "exclude_preprocessors cannot be used together.")

        if dataset_properties is None or not isinstance(dataset_properties, dict):
            dataset_properties = dict()

        if 'sparse' not in dataset_properties:
            # This dataset is probaby dense
            dataset_properties['sparse'] = False

        # Compile a list of legal preprocessors for this problem
        available_preprocessors = components.preprocessing_components._preprocessors
        preprocessors = ParamSklearnClassifier.get_available_components(
            available_comp=available_preprocessors,
            data_prop=dataset_properties,
            inc=include_preprocessors,
            exc=exclude_preprocessors)

        # Compile a list of all estimator objects for this problem
        available_classifiers = ParamSklearnClassifier._get_estimator_components()
        classifiers = ParamSklearnClassifier.get_available_components(
            available_comp=available_classifiers,
            data_prop=dataset_properties,
            inc=include_estimators,
            exc=exclude_estimators)

        if len(classifiers) == 0:
            raise ValueError("No classifiers found")
        if len(preprocessors) == 0:
            raise ValueError("No preprocessors found, please add NoPreprocessing")

        preprocessors_list = preprocessors.keys()
        classifiers_list = classifiers.keys()
        matches = ParamSklearn.create_searchspace_util.get_match_array(
            preprocessors=preprocessors, estimators=classifiers,
            sparse=dataset_properties.get('sparse'), pipeline=cls._get_pipeline())

        # Now we have only legal preprocessors/classifiers we combine them
        # Simple sanity checks
        assert np.sum(matches) != 0, "No valid preprocessor/classifier " \
                                     "combination found, probably a bug"
        assert np.sum(matches) <= (matches.shape[0] * matches.shape[1]), \
            "'matches' is not binary; %s <= %d, [%d*%d]" % \
            (str(np.sum(matches)), matches.shape[0]*matches.shape[1],
             matches.shape[0], matches.shape[1])

        if np.sum(matches) < (matches.shape[0] * matches.shape[1]):
            matches, preprocessors_list, classifiers_list, preprocessors, classifiers = \
                ParamSklearn.create_searchspace_util.sanitize_arrays(
                    m=matches, preprocessors_list=preprocessors_list,
                    estimators_list=classifiers_list,
                    preprocessors=preprocessors, estimators=classifiers)

        # Sanity checks
        assert len(preprocessors_list) > 0, "No valid preprocessors found"
        assert len(classifiers_list) > 0, "No valid classifiers found"

        assert len(preprocessors_list) == matches.shape[0], \
            "Preprocessor deleting went wrong"
        assert len(classifiers_list) == matches.shape[1], \
            "Classifier deleting went wrong"
        assert [c in classifiers_list for c in classifiers]
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
        classifier_defaults = ['random_forest', 'liblinear', 'sgd',
                               'libsvm_svc']
        classifier_default = None
        for cd_ in classifier_defaults:
            # Make sure that a classifier which can only handle dense is not
            # selected as the default for a sparse dataset
            if cd_ not in classifiers:
                continue
            no_preprocessing_idx = preprocessors_list.index(preprocessor_default)
            cd_index = classifiers_list.index(cd_)
            if matches[no_preprocessing_idx, cd_index] == 1:
                classifier_default = cd_
                break
        if classifier_default is None:
            classifier_default = classifiers.keys()[0]

        # Get the configuration space
        configuration_space = super(ParamSklearnClassifier, cls).\
            get_hyperparameter_search_space(estimator_name=cls._get_estimator_hyperparameter_name(),
                                            default_estimator=classifier_default,
                                            estimator_components=classifiers,
                                            default_preprocessor=preprocessor_default,
                                            preprocessor_components=preprocessors,
                                            dataset_properties=dataset_properties,
                                            always_active=cls._get_pipeline())

        # And now add forbidden parameter configurations
        # According to matches
        configuration_space = ParamSklearn.create_searchspace_util.add_forbidden(
            conf_space=configuration_space, preproc_list=preprocessors_list,
            est_list=classifiers_list, matches=matches, est_type="classifier")

        # A classifier which can handle sparse data after the densifier
        for key in classifiers:
            if SPARSE in classifiers[key].get_properties()['input']:
                try:
                    configuration_space.add_forbidden_clause(
                        ForbiddenAndConjunction(
                            ForbiddenEqualsClause(
                                configuration_space.get_hyperparameter(
                                    'classifier'), key),
                            ForbiddenEqualsClause(
                                configuration_space.get_hyperparameter(
                                    'preprocessor'), 'densifier')
                        ))
                except ValueError as e:
                    if e.message.startswith("Forbidden clause must be "
                                            "instantiated with a legal "
                                            "hyperparameter value for "
                                            "'preprocessor"):
                        pass
                    else:
                        raise e

        # which would take too long
        # Combinations of non-linear models with feature learning:
        classifiers_ = ["adaboost", "extra_trees", "gradient_boosting",
                        "k_nearest_neighbors", "libsvm_svc", "random_forest",
                        "gaussian_nb", "gaussian_process", "decision_tree"]
        feature_learning = ["kitchen_sinks", "nystroem_sampler", "dictionary_learning"]

        for c, f in product(classifiers_, feature_learning):
            if c not in classifiers_list:
                continue
            if f not in preprocessors_list:
                continue
            try:
                configuration_space.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "classifier"), c),
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "preprocessor"), f)))
            except KeyError:
                pass

        # We have seen empirically that tree-based models together with PCA
        # don't work better than tree-based models without preprocessing
        classifiers_ = ["random_forest", "extra_trees", "gradient_boosting",
                        "decision_tree"]
        for c in classifiers_:
            if c not in classifiers_list:
                continue
            try:
                configuration_space.add_forbidden_clause(
                    ForbiddenAndConjunction(
                        ForbiddenEqualsClause(
                            configuration_space.get_hyperparameter(
                                "preprocessor"), "pca"),
                        ForbiddenEqualsClause(
                            configuration_space.get_hyperparameter(
                                "classifier"), c)))
            except KeyError:
                pass
            except ValueError as e:
                if e.message.startswith("Forbidden clause must be "
                                        "instantiated with a legal "
                                        "hyperparameter value for "
                                        "'preprocessor"):
                    pass
                else:
                    raise e

        # Won't work
        # Multinomial NB does not work with negative values, don't use
        # it with standardization, features learning, pca
        classifiers_ = ["multinomial_nb", "bernoulli_nb"]
        preproc_with_negative_X = ["kitchen_sinks", "pca", "truncatedSVD",
                                   "fast_ica", "kernel_pca", "nystroem_sampler"]
        scaling_strategies = ['standard', 'none']
        for c in classifiers_:
            if c not in classifiers_list:
                continue
            for scaling_strategy in scaling_strategies:
                try:
                    configuration_space.add_forbidden_clause(ForbiddenAndConjunction(
                        ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                            "rescaling:strategy"), scaling_strategy),
                        ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                            "classifier"), c)))
                except KeyError:
                    pass

        for c, f in product(classifiers_, preproc_with_negative_X):
            if c not in classifiers_list:
                continue
            if f not in preprocessors_list:
                continue
            try:
                configuration_space.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "preprocessor"), f),
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "classifier"), c)))
            except KeyError:
                pass

        # Now try to add things for which we know that they don't work
        forbidden_hyperparameter_combinations = \
            [("select_percentile_classification:score_func", "chi2",
              "rescaling:strategy", "standard"),
             ("select_percentile_classification:score_func", "chi2",
              "rescaling:strategy", "none"),
             ("select_rates:score_func", "chi2",
              "rescaling:strategy", "standard"),
             ("select_rates:score_func", "chi2",
              "rescaling:strategy", "none"),
             ("nystroem_sampler:kernel", 'chi2', "rescaling:strategy",
              "standard"),
             ("nystroem_sampler:kernel", 'chi2', "rescaling:strategy",
              "none")]
        for hp_name_1, hp_value_1, hp_name_2, hp_value_2 in \
                forbidden_hyperparameter_combinations:
            try:
                configuration_space.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        hp_name_1), hp_value_1),
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        hp_name_2), hp_value_2)
                ))
            except:
                pass

        return configuration_space

    @staticmethod
    def _get_estimator_hyperparameter_name():
        return "classifier"

    @staticmethod
    def _get_estimator_components():
        return components.classification_components._classifiers

    @staticmethod
    def _get_pipeline():
        return ["imputation", "rescaling", "balancing", "__preprocessor__",
                "__estimator__"]