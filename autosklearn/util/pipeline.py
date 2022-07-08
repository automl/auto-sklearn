# -*- encoding: utf-8 -*-
from typing import Dict, List, Optional, Union

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace

from autosklearn.constants import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    MULTILABEL_CLASSIFICATION,
    MULTIOUTPUT_REGRESSION,
    REGRESSION_TASKS,
)
from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from autosklearn.pipeline.regression import SimpleRegressionPipeline

__all__ = ["get_configuration_space"]


def get_configuration_space(
    datamanager: AbstractDataManager,
    include: Optional[Dict[str, List[str]]] = None,
    exclude: Optional[Dict[str, List[str]]] = None,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> ConfigurationSpace:
    """Get the configuration of a pipeline given some dataset info

    Parameters
    ----------
    datamanager: AbstractDataManager
        AbstractDataManager object storing all important information about the dataset

    include: Optional[Dict[str, List[str]]] = None
        A dictionary of what components to include for each pipeline step

    exclude: Optional[Dict[str, List[str]]] = None
        A dictionary of what components to exclude for each pipeline step

    random_state: Optional[Union[int, np.random.Randomstate]] = None
        The random state to use for seeding the ConfigSpace

    Returns
    -------
    ConfigurationSpace
        The configuration space for the pipeline
    """
    if datamanager.info["task"] in REGRESSION_TASKS:
        return _get_regression_configuration_space(
            datamanager, include, exclude, random_state
        )
    else:
        return _get_classification_configuration_space(
            datamanager, include, exclude, random_state
        )


def _get_regression_configuration_space(
    datamanager: AbstractDataManager,
    include: Optional[Dict[str, List[str]]],
    exclude: Optional[Dict[str, List[str]]],
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> ConfigurationSpace:
    """Get the configuration of a regression pipeline given some dataset info

    Parameters
    ----------
    datamanager: AbstractDataManager
        AbstractDataManager object storing all important information about the dataset

    include: Optional[Dict[str, List[str]]] = None
        A dictionary of what components to include for each pipeline step

    exclude: Optional[Dict[str, List[str]]] = None
        A dictionary of what components to exclude for each pipeline step

    random_state: Optional[Union[int, np.random.Randomstate]] = None
        The random state to use for seeding the ConfigSpace

    Returns
    -------
    ConfigurationSpace
        The configuration space for the regression pipeline
    """
    task_type = datamanager.info["task"]
    sparse = False
    multioutput = False
    if task_type == MULTIOUTPUT_REGRESSION:
        multioutput = True

    if datamanager.info["is_sparse"] == 1:
        sparse = True

    dataset_properties = {"multioutput": multioutput, "sparse": sparse}

    configuration_space = SimpleRegressionPipeline(
        feat_type=datamanager.feat_type,
        dataset_properties=dataset_properties,
        include=include,
        exclude=exclude,
        random_state=random_state,
    ).get_hyperparameter_search_space(feat_type=datamanager.feat_type)
    return configuration_space


def _get_classification_configuration_space(
    datamanager: AbstractDataManager,
    include: Optional[Dict[str, List[str]]],
    exclude: Optional[Dict[str, List[str]]],
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> ConfigurationSpace:
    """Get the configuration of a classification pipeline given some dataset info

    Parameters
    ----------
    datamanager: AbstractDataManager
         AbstractDataManager object storing all important information about the dataset

    include: Optional[Dict[str, List[str]]] = None
        A dictionary of what components to include for each pipeline step

    exclude: Optional[Dict[str, List[str]]] = None
        A dictionary of what components to exclude for each pipeline step

    random_state: Optional[Union[int, np.random.Randomstate]] = None
        The random state to use for seeding the ConfigSpace

    Returns
    -------
    ConfigurationSpace
        The configuration space for the classification pipeline
    """
    task_type = datamanager.info["task"]

    multilabel = False
    multiclass = False
    sparse = False

    if task_type == MULTILABEL_CLASSIFICATION:
        multilabel = True
    if task_type == MULTICLASS_CLASSIFICATION:
        multiclass = True
    if task_type == BINARY_CLASSIFICATION:
        pass

    if datamanager.info["is_sparse"] == 1:
        sparse = True

    dataset_properties = {
        "multilabel": multilabel,
        "multiclass": multiclass,
        "sparse": sparse,
    }

    return SimpleClassificationPipeline(
        feat_type=datamanager.feat_type,
        dataset_properties=dataset_properties,
        include=include,
        exclude=exclude,
        random_state=random_state,
    ).get_hyperparameter_search_space(feat_type=datamanager.feat_type)
