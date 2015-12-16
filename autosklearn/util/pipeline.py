# -*- encoding: utf-8 -*-
from autosklearn.constants import *
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from autosklearn.pipeline.regression import SimpleRegressionPipeline


__all__ = [
    'get_configuration_space',
    'get_class',
    'get_model'
]


def get_configuration_space(info,
                            include_estimators=None,
                            include_preprocessors=None):
    include = dict()
    if include_preprocessors is not None:
        include['preprocessor'] = include_preprocessors
    if info['task'] in REGRESSION_TASKS:
        if include_estimators is not None:
            include['regressor'] = include_estimators
        return _get_regression_configuration_space(info, include)
    else:
        if include_estimators is not None:
            include['classifier'] = include_estimators
        return _get_classification_configuration_space(info, include)


def _get_regression_configuration_space(info, include):
    sparse = False
    if info['is_sparse'] == 1:
        sparse = True
    configuration_space = SimpleRegressionPipeline. \
        get_hyperparameter_search_space(include=include,
                                        dataset_properties={'sparse': sparse})
    return configuration_space


def _get_classification_configuration_space(info, include):
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
    if task_type == BINARY_CLASSIFICATION:
        pass

    if info['is_sparse'] == 1:
        sparse = True

    dataset_properties = {
        'multilabel': multilabel,
        'multiclass': multiclass,
        'sparse': sparse
    }

    return SimpleClassificationPipeline.get_hyperparameter_search_space(
        dataset_properties=dataset_properties,
        include=include)


def get_model(configuration, seed):
    if 'classifier' in configuration:
        return SimpleClassificationPipeline(configuration, seed)
    elif 'regressor' in configuration:
        return SimpleRegressionPipeline(configuration, seed)


def get_class(info):
    if info['task'] in REGRESSION_TASKS:
        return SimpleRegressionPipeline
    else:
        return SimpleClassificationPipeline
