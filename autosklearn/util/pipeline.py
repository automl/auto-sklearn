# -*- encoding: utf-8 -*-
from autosklearn.constants import *
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from autosklearn.pipeline.regression import SimpleRegressionPipeline


__all__ = [
    'get_configuration_space',
    'get_class',
]


def get_configuration_space(info,
                            include_estimators=None,
                            exclude_estimators=None,
                            include_preprocessors=None,
                            exclude_preprocessors=None):
    exclude = dict()
    include = dict()
    if include_preprocessors is not None and \
            exclude_preprocessors is not None:
        raise ValueError('Cannot specify include_preprocessors and '
                         'exclude_preprocessors.')
    elif include_preprocessors is not None:
        include['preprocessor'] = include_preprocessors
    elif exclude_preprocessors is not None:
        exclude['preprocessor'] = exclude_preprocessors

    if include_estimators is not None and \
            exclude_estimators is not None:
        raise ValueError('Cannot specify include_estimators and '
                         'exclude_estimators.')
    elif include_estimators is not None:
        if info['task'] in CLASSIFICATION_TASKS:
            include['classifier'] = include_estimators
        elif info['task'] in REGRESSION_TASKS:
            include['regressor'] = include_estimators
        else:
            raise ValueError(info['task'])
    elif exclude_estimators is not None:
        if info['task'] in CLASSIFICATION_TASKS:
            exclude['classifier'] = exclude_estimators
        elif info['task'] in REGRESSION_TASKS:
            exclude['regressor'] = exclude_estimators
        else:
            raise ValueError(info['task'])

    if info['task'] in REGRESSION_TASKS:
        return _get_regression_configuration_space(info, include, exclude)
    else:
        return _get_classification_configuration_space(info, include, exclude)


def _get_regression_configuration_space(info, include, exclude):
    sparse = False
    if info['is_sparse'] == 1:
        sparse = True
    configuration_space = SimpleRegressionPipeline(
        dataset_properties={'sparse': sparse},
        include=include,
        exclude=exclude
    ).get_hyperparameter_search_space()
    return configuration_space


def _get_classification_configuration_space(info, include, exclude):
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

    return SimpleClassificationPipeline(
        dataset_properties=dataset_properties,
        include=include, exclude=exclude).\
        get_hyperparameter_search_space()


def get_class(info):
    if info['task'] in REGRESSION_TASKS:
        return SimpleRegressionPipeline
    else:
        return SimpleClassificationPipeline
