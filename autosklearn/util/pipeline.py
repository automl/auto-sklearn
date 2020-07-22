# -*- encoding: utf-8 -*-
from typing import Any, Dict, List, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

from sklearn.pipeline import Pipeline

from autosklearn.constants import (
    BINARY_CLASSIFICATION,
    CLASSIFICATION_TASKS,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
    MULTIOUTPUT_REGRESSION,
    REGRESSION,
    REGRESSION_TASKS
)
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from autosklearn.pipeline.regression import SimpleRegressionPipeline


__all__ = [
    'get_configuration_space',
    'get_class',
]


def get_configuration_space(info: Dict[str, Any],
                            include_estimators: Optional[List[str]] = None,
                            exclude_estimators: Optional[List[str]] = None,
                            include_preprocessors: Optional[List[str]] = None,
                            exclude_preprocessors: Optional[List[str]] = None
                            ) -> ConfigurationSpace:
    exclude = dict()
    include = dict()
    if include_preprocessors is not None and \
            exclude_preprocessors is not None:
        raise ValueError('Cannot specify include_preprocessors and '
                         'exclude_preprocessors.')
    elif include_preprocessors is not None:
        include['feature_preprocessor'] = include_preprocessors
    elif exclude_preprocessors is not None:
        exclude['feature_preprocessor'] = exclude_preprocessors

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


def _get_regression_configuration_space(info: Dict[str, Any], include: Dict[str, List[str]],
                                        exclude: Dict[str, List[str]]) -> ConfigurationSpace:
    task_type = info['task']
    sparse = False
    multioutput = False
    if task_type == MULTIOUTPUT_REGRESSION:
        multioutput = True

    dataset_properties = {
        'multioutput': multioutput,
        'sparse': sparse
    }

    if info['is_sparse'] == 1:
        sparse = True
    configuration_space = SimpleRegressionPipeline(
        dataset_properties=dataset_properties,
        include=include,
        exclude=exclude
    ).get_hyperparameter_search_space()
    return configuration_space


def _get_classification_configuration_space(info: Dict[str, Any], include: Dict[str, List[str]],
                                            exclude: Dict[str, List[str]]) -> ConfigurationSpace:
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


def get_class(info: Dict[str, Any]) -> Pipeline:
    if info['task'] in REGRESSION_TASKS:
        return SimpleRegressionPipeline
    else:
        return SimpleClassificationPipeline
