# -*- encoding: utf-8 -*-
from typing import Any, Dict, List, Optional

from ConfigSpace.configuration_space import ConfigurationSpace

from sklearn.pipeline import Pipeline

from autosklearn.constants import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
    MULTIOUTPUT_REGRESSION,
    REGRESSION_TASKS
)
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from autosklearn.pipeline.regression import SimpleRegressionPipeline


__all__ = [
    'get_configuration_space',
    'get_class',
]


def get_configuration_space(info: Dict[str, Any],
                            include: Optional[Dict[str, List[str]]] = None,
                            exclude: Optional[Dict[str, List[str]]] = None,
                            ) -> ConfigurationSpace:

    if info['task'] in REGRESSION_TASKS:
        return _get_regression_configuration_space(info, include, exclude)
    else:
        return _get_classification_configuration_space(info, include, exclude)


def _get_regression_configuration_space(info: Dict[str, Any],
                                        include: Optional[Dict[str, List[str]]],
                                        exclude: Optional[Dict[str, List[str]]]
                                        ) -> ConfigurationSpace:
    task_type = info['task']
    sparse = False
    multioutput = False
    if task_type == MULTIOUTPUT_REGRESSION:
        multioutput = True

    if info['is_sparse'] == 1:
        sparse = True

    dataset_properties = {
        'multioutput': multioutput,
        'sparse': sparse
    }

    configuration_space = SimpleRegressionPipeline(
        dataset_properties=dataset_properties,
        include=include,
        exclude=exclude
    ).get_hyperparameter_search_space()
    return configuration_space


def _get_classification_configuration_space(info: Dict[str, Any],
                                            include: Optional[Dict[str, List[str]]],
                                            exclude: Optional[Dict[str, List[str]]]
                                            ) -> ConfigurationSpace:
    task_type = info['task']

    multilabel = False
    multiclass = False
    sparse = False

    if task_type == MULTILABEL_CLASSIFICATION:
        multilabel = True
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
