from collections import OrderedDict
from itertools import product

import numpy as np

from sklearn.base import ClassifierMixin

from HPOlibConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction

from ParamSklearn import components as components
from ParamSklearn.base import ParamSklearnBaseEstimator
from ParamSklearn.util import SPARSE, DENSE, INPUT


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

    def predict_proba(self, X):
        """predict_proba.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
        """
        self._validate_input_X(X)
        Xt = X
        for name, transform in self._pipeline.steps[:-1]:
            Xt = transform.transform(Xt)

        return self._pipeline.steps[-1][-1].predict_proba(Xt)

    @classmethod
    def create_match_array(cls, preprocessors, classifiers, sparse):
        # Now select combinations that work
        # We build a binary matrix, where a 1 indicates, that a combination
        # work on this dataset based in the dataset and the input/output formats
        # A 'zero'-row (column) is an unusable preprocessor (classifier)
        # A single zero results in an forbidden condition
        preprocessors_list = preprocessors.keys()
        classifiers_list = classifiers.keys()
        matches = np.zeros([len(preprocessors), len(classifiers)])
        for pidx, p in enumerate(preprocessors_list):
            p_out = preprocessors[p].get_properties()['output']
            for cidx, c in enumerate(classifiers_list):
                c_in = classifiers[c].get_properties()['input']
                if p_out == INPUT:
                    # Preprocessor does not change the format
                    if (sparse and SPARSE in c_in) or \
                            (not sparse and DENSE in c_in):
                        # Classifier input = Dataset format
                        matches[pidx, cidx] = 1
                        continue
                    else:
                        # These won't work
                        pass
                elif p_out == DENSE and DENSE in c_in:
                    matches[pidx, cidx] = 1
                    continue
                elif p_out == SPARSE and SPARSE in c_in:
                    matches[pidx, cidx] = 1
                    continue
                else:
                    # These won't work
                    pass
        return matches, preprocessors_list, classifiers_list

    @classmethod
    def remove_non_matches(cls, matches, preprocessors_list, classifiers_list):
        # We might delete some rows/columns
        l = len(preprocessors_list)
        for pidx, p in enumerate(preprocessors_list):
            # We use the reverse idx as it stays correct
            # when we start removing rows
            reverse_idx = -l + pidx
            if (matches[pidx, :] == 0).all():
                # unusable preprocessor, delete row
                matches = np.delete(matches, reverse_idx, axis=0)
                #del preprocessors[p]
                del preprocessors_list[reverse_idx]
        l = len(classifiers_list)
        for cidx, c in enumerate(classifiers_list):
            # We use the reverse idx as it stays correct
            # when we start removing cols
            reverse_idx = -l + cidx
            if (matches[:, cidx] == 0).all():
                # unusable preprocessor, delete row
                matches = np.delete(matches, reverse_idx, axis=1)
                #del classifiers[c]
                del classifiers_list[reverse_idx]
        return matches, preprocessors_list, classifiers_list

    @classmethod
    def add_forbidden_clauses(cls, configuration_space, preprocessors_list, classifiers_list, matches):
        for pdx, p in enumerate(preprocessors_list):
            if np.sum(matches[pdx, :]) == matches.shape[1]:
                continue
            for cdx, c in enumerate(classifiers_list):
                if matches[pdx, cdx] == 0:
                    try:
                        configuration_space.add_forbidden_clause(ForbiddenAndConjunction(
                            ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                                "classifier"), c),
                            ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                                "preprocessor"), p)))
                    except:
                        pass
        return configuration_space

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
            dataset_properties = {}
        if 'sparse' not in dataset_properties:
            # This dataset is probaby dense
            dataset_properties['sparse'] = False

        # Compile a list of legal preprocessors for this problem
        available_preprocessors = \
            components.preprocessing_components._preprocessors
        preprocessors = OrderedDict()
        for name in available_preprocessors:
            if name in cls._get_pipeline():
                # We don't want these preprocessors, as they are always included
                # preprocessors[name] = available_preprocessors[name]
                continue
            elif include_preprocessors is not None and \
                            name not in include_preprocessors:
                continue
            elif exclude_preprocessors is not None and \
                            name in exclude_preprocessors:
                continue

            if available_preprocessors[name]. \
                    get_properties()['handles_classification'] is False:
                continue
            if dataset_properties.get('multiclass') is True and \
                    available_preprocessors[name].get_properties()[
                                'handles_multiclass'] is False:
                continue
            if dataset_properties.get('multilabel') is True and \
                    available_preprocessors[name].get_properties()[
                                'handles_multilabel'] is False:
                continue
            if dataset_properties.get('sparse') is True and \
                    SPARSE not in available_preprocessors[name].get_properties()['input']:
                continue
            elif dataset_properties.get('sparse') is False and \
                    DENSE not in available_preprocessors[name].get_properties()['input']:
                continue

            preprocessors[name] = available_preprocessors[name]

        # Compile a list of all estimator objects for this problem
        available_classifiers = ParamSklearnClassifier._get_estimator_components()

        # Remove unwanted classifiers
        classifiers = OrderedDict()
        for name in available_classifiers:
            if include_estimators is not None and name not in include_estimators:
                continue
            elif exclude_estimators is not None and name in exclude_estimators:
                continue

            if dataset_properties.get('multiclass') is True and \
                    available_classifiers[name].get_properties()[
                        'handles_multiclass'] is False:
                continue
            if dataset_properties.get('multilabel') is True and \
                    available_classifiers[name].get_properties()[
                        'handles_multilabel'] is False:
                continue
            classifiers[name] = available_classifiers[name]
        if len(classifiers) == 0:
            raise ValueError("No classifier to build a configuration space "
                             "for...")

        matches, preprocessors_list, classifiers_list = ParamSklearnClassifier.\
            create_match_array(preprocessors=preprocessors,
                               classifiers=classifiers,
                               sparse=dataset_properties.get('sparse'))

        # Now we have only legal preprocessors/classifiers we combine them
        # Simple sanity checks
        assert np.sum(matches) != 0, "No valid preprocessor/classifier " \
                                     "combination found, this might be a bug"
        assert np.sum(matches) <= (matches.shape[0] * matches.shape[1]), \
            "'matches' is not binary; %s <= %d, [%d*%d]" % \
            (str(np.sum(matches)), matches.shape[0]*matches.shape[1],
             matches.shape[0], matches.shape[1])

        if np.sum(matches) < (matches.shape[0] * matches.shape[1]):
            matches, preprocessors_list, classifiers_list = ParamSklearnClassifier.\
                remove_non_matches(matches=matches,
                                   preprocessors_list=preprocessors_list,
                                   classifiers_list=classifiers_list)
            for p in preprocessors.keys():
                if p not in preprocessors_list:
                    del preprocessors[p]
            for c in classifiers.keys():
                if c not in classifiers_list:
                    del classifiers[c]

        # Sanity checks
        assert len(preprocessors_list) == matches.shape[0], \
            "Preprocessor deleting went wrong"
        assert len(classifiers_list) == matches.shape[1], \
            "Classifier deleting went wrong"
        assert [c in classifiers_list for c in classifiers]
        assert [p in preprocessors_list for p in preprocessors]

        # Now add always present preprocessors
        for name in available_preprocessors:
            if name in cls._get_pipeline():
                preprocessors[name] = available_preprocessors[name]

        # Hardcode the defaults based on some educated guesses
        classifier_defaults = ['random_forest', 'liblinear', 'sgd',
                               'libsvm_svc']
        classifier_default = None
        for cd_ in classifier_defaults:
            if cd_ in classifiers:
                classifier_default = cd_
                break
        if classifier_default is None:
            classifier_default = classifiers.keys()[0]

        # Get the configuration space
        configuration_space = super(ParamSklearnClassifier, cls)\
            .get_hyperparameter_search_space(
            cls._get_estimator_hyperparameter_name(),
            classifier_default, classifiers, preprocessors, dataset_properties,
            cls._get_pipeline())

        # And now add forbidden parameter configurations
        # According to matches
        configuration_space = ParamSklearnClassifier.add_forbidden_clauses(
            configuration_space=configuration_space,
            preprocessors_list=preprocessors_list,
            classifiers_list=classifiers_list, matches=matches)

        # which would take too long
        # Combinations of tree-based models with feature learning:
        classifiers_ = ["extra_trees", "gradient_boosting",
                        "k_nearest_neighbors", "libsvm_svc", "random_forest"]
        feature_learning_ = ["kitchen_sinks", "sparse_filtering"]

        for c, f in product(classifiers_, feature_learning_):
            try:
                configuration_space.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "classifier"), c),
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "preprocessor"), f)))
            except:
                pass

        # Won't work
        # Multinomial NB does not work with negative values, don't use
        # it with standardization, features learning, pca
        classifiers_ = ["multinomial_nb", "bagged_multinomial_nb",
                       "bernoulli_nb"]
        feature_learning_ = ["kitchen_sinks", "sparse_filtering", "pca"]
        for c in classifiers_:
            if c not in classifiers_list:
                continue
            try:
                configuration_space.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "rescaling:strategy"), "standard"),
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "classifier"), c)))
            except:
                pass

        for c, f in product(classifiers_, feature_learning_):
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
            except:
                pass

        return configuration_space


        """
        # Compile a list of all estimator objects for this problem
        available_classifiers = ParamSklearnClassifier._get_estimator_components()

        classifiers = dict()
        for name in available_classifiers:
            if include_estimators is not None and \
                            name not in include_estimators:
                continue
            elif exclude_estimators is not None and \
                            name in exclude_estimators:
                continue

            if dataset_properties.get('multiclass') is True and \
                    available_classifiers[name].get_properties()[
                        'handles_multiclass'] is False:
                continue
            if dataset_properties.get('multilabel') is True and \
                    available_classifiers[name].get_properties()[
                        'handles_multilabel'] is False:
                continue
            if dataset_properties.get('sparse') is True and \
                    available_classifiers[name].get_properties()[
                        'handles_sparse'] is False:
                continue
            classifiers[name] = available_classifiers[name]

        if len(classifiers) == 0:
            raise ValueError("No classifier to build a configuration space "
                             "for...")

        # Hardcode the defaults based on some educated guesses
        classifier_defaults = ['random_forest', 'liblinear', 'sgd',
                               'libsvm_svc']
        classifier_default = None
        for cd_ in classifier_defaults:
            if cd_ in classifiers:
                classifier_default = cd_
                break
        if classifier_default is None:
            classifier_default = classifiers.keys()[0]

        # Compile a list of preprocessor for this problem
        available_preprocessors = \
            components.preprocessing_components._preprocessors

        preprocessors = dict()
        for name in available_preprocessors:
            if name in cls._get_pipeline():
                preprocessors[name] = available_preprocessors[name]
                continue
            elif include_preprocessors is not None and \
                            name not in include_preprocessors:
                continue
            elif exclude_preprocessors is not None and \
                            name in exclude_preprocessors:
                continue

            if available_preprocessors[name]. \
                    get_properties()['handles_classification'] is False:
                continue
            if dataset_properties.get('multiclass') is True and \
                    available_preprocessors[name].get_properties()[
                                'handles_multiclass'] is False:
                continue
            if dataset_properties.get('multilabel') is True and \
                    available_preprocessors[name].get_properties()[
                                'handles_multilabel'] is False:
                continue
            if dataset_properties.get('sparse') is True and \
                    available_preprocessors[name].get_properties()[
                                'handles_sparse'] is False:
                continue
            elif dataset_properties.get('sparse') is False and \
                    available_preprocessors[name].get_properties()[
                                'handles_dense'] is False:
                continue

            preprocessors[name] = available_preprocessors[name]

        # Get the configuration space
        configuration_space = super(ParamSklearnClassifier, cls)\
            .get_hyperparameter_search_space(
            cls._get_estimator_hyperparameter_name(),
            classifier_default, classifiers, preprocessors, dataset_properties,
            cls._get_pipeline())

        # And now add forbidden parameter configurations which would take too
        # long

        # Combinations of tree-based models with feature learning:
        classifiers_ = ["extra_trees", "gradient_boosting",
                        "k_nearest_neighbors", "libsvm_svc", "random_forest"]
        feature_learning_ = ["kitchen_sinks", "sparse_filtering"]

        for c, f in product(classifiers_, feature_learning_):
            try:
                configuration_space.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "classifier"), c),
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "preprocessor"), f)))
            except:
                pass

        # Multinomial NB does not work with negative values -> so don't use
        # it with standardization, features learning, pca
        classifiers_ = ["multinomial_nb", "bagged_multinomial_nb",
                       "bernoulli_nb"]
        feature_learning_ = ["kitchen_sinks", "sparse_filtering", "pca"]
        for c in classifiers_:
            try:
                configuration_space.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "rescaling:strategy"), "standard"),
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "classifier"), c)))
            except:
                pass
        for c, f in product(classifiers_, feature_learning_):
            try:
                configuration_space.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "preprocessor"), f),
                    ForbiddenEqualsClause(configuration_space.get_hyperparameter(
                        "classifier"), c)))
            except:
                pass

        return configuration_space
        """

    @staticmethod
    def _get_estimator_hyperparameter_name():
        return "classifier"

    @staticmethod
    def _get_estimator_components():
        return components.classification_components._classifiers

    @staticmethod
    def _get_pipeline():
        return ["imputation", "rescaling", "__preprocessor__", "__estimator__"]