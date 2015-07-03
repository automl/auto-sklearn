from ParamSklearn.classification import ParamSklearnClassifier
from ParamSklearn.regression import ParamSklearnRegressor

from autosklearn.constants import *


def get_configuration_space(info, include_estimators=None,
                            include_preprocessors=None):
    if info['task'] in REGRESSION_TASKS:
        return _get_regression_configuration_space(info, include_estimators,
                                                   include_preprocessors)
    else:
        return _get_classification_configuration_space(info,
                                                       include_estimators,
                                                       include_preprocessors)


def _get_regression_configuration_space(info, include_estimators=None,
                                        include_preprocessors=None):
    sparse = False
    if info['is_sparse'] == 1:
        sparse = True
    configuration_space = ParamSklearnRegressor. \
        get_hyperparameter_search_space(include_estimators=include_estimators,
                                        include_preprocessors=include_preprocessors,
                                        dataset_properties={'sparse': sparse})
    return configuration_space


def _get_classification_configuration_space(info, include_estimators=None,
                                            include_preprocessors=None):
    task_type = info['task']

    multilabel = False
    multiclass = False
    sparse = False

    if task_type == MULTILABEL_CLASSIFICATION:
        multilabel = True
    if task_type == REGRESSION:
        raise NotImplementedError()
    if task_type == MULTICLASS_CLASSIFICATION:
        multiclass = True
        pass
    if task_type == BINARY_CLASSIFICATION:
        pass

    if info['is_sparse'] == 1:
        sparse = True

    dataset_properties = {'multilabel': multilabel, 'multiclass': multiclass,
                          'sparse': sparse}

    return ParamSklearnClassifier.get_hyperparameter_search_space(
        dataset_properties=dataset_properties,
        include_estimators=include_estimators,
        include_preprocessors=include_preprocessors)
        # exclude_preprocessors=["sparse_filtering"])


def get_model(configuration, seed):
    if 'classifier' in configuration:
        return ParamSklearnClassifier(configuration, seed)
    elif 'regressor' in configuration:
        return  ParamSklearnRegressor(configuration, seed)


def get_class(info):
    if info['task'] in REGRESSION_TASKS:
        return ParamSklearnRegressor
    else:
        return ParamSklearnClassifier
