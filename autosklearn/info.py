"""
This module servers as an introspection point for things users might
want to programatically query about autosklearn.
"""
from __future__ import annotations

from typing import Any, Generic, Type, TypeVar

from dataclasses import dataclass

from typing_extensions import Literal

from autosklearn.pipeline.components.base import (
    AutoSklearnClassificationAlgorithm,
    AutoSklearnComponent,
    AutoSklearnPreprocessingAlgorithm,
    AutoSklearnRegressionAlgorithm,
)
from autosklearn.pipeline.components.classification import ClassifierChoice
from autosklearn.pipeline.components.data_preprocessing import DataPreprocessorChoice
from autosklearn.pipeline.components.feature_preprocessing import (
    FeaturePreprocessorChoice,
)
from autosklearn.pipeline.components.regression import RegressorChoice
from autosklearn.pipeline.constants import DATASET_PROPERTIES_TO_STRING

# Something that is a type that inherits from AutoSklearnComponent
T = TypeVar("T", bound=Type[AutoSklearnComponent])


def _translate_properties(
    props: dict[str, Any],
    kind: Literal["classifier", "regressor", "f_preprocessor", "d_preprocessor"],
) -> dict[str, Any]:
    """Converts supported inputs and outputs to strings"""
    # This is information is conveyed implicitly by being a regressor/classifier ...
    delwords = ["handles_regression", "handles_classification"]

    # Covered by input type, duplicated info
    delwords += ["handles_sparse", "handles_dense"]

    # Words we rename (from, to)
    popwords: list[tuple[str, str]] = [
        ("input", "supported_inputs"),
        ("output", "output_kind"),
        ("is_deterministic", "deterministic"),
    ]

    if kind in ["classifier", "f_preprocessor", "d_preprocessor"]:
        delwords += ["handles_multioutput"]

    if kind in ["regressor", "f_preprocessor", "d_preprocessor"]:
        delwords += ["handles_multiclass", "handles_multilabel"]

    for word in delwords:
        if word in props:
            del props[word]

    for frm, to in popwords:
        props[to] = props.pop(frm)

    props["supported_inputs"] = [
        DATASET_PROPERTIES_TO_STRING[k] for k in props["supported_inputs"]
    ]
    props["output_kind"] = DATASET_PROPERTIES_TO_STRING[props["output_kind"][0]]

    return props


@dataclass
class _ComponentInfo(Generic[T]):
    type: T  # cls is not possible due to @dataclass conversion
    name: str
    shortname: str
    output_kind: str
    supported_inputs: list[str]
    deterministic: bool = False


@dataclass
class RegressorInfo(_ComponentInfo[Type[AutoSklearnRegressionAlgorithm]]):
    handles_multioutput: bool = False
    prefers_data_normalized: bool = False


@dataclass
class ClassifierInfo(_ComponentInfo[Type[AutoSklearnClassificationAlgorithm]]):
    handles_binary: bool = True  # We assume all components support this
    handles_multiclass: bool = False
    handles_multilabel: bool = False
    handles_multilabel_multiclass = False


@dataclass
class FeaturePreprocessorInfo(_ComponentInfo[Type[AutoSklearnPreprocessingAlgorithm]]):
    pass


@dataclass
class DataPreprocessorInfo(_ComponentInfo[Type[AutoSklearnPreprocessingAlgorithm]]):
    # There should be more here but our DataPreprocessing part of the pipeline doesn't
    # pick up on it because there's on FeatTypeSplit available which further has
    # subcomponents with extra properties
    pass


@dataclass
class ComponentsInfo:
    classifiers: dict[str, ClassifierInfo]
    regressors: dict[str, RegressorInfo]
    feature_preprocessors: dict[str, FeaturePreprocessorInfo]
    data_preprocessors: dict[str, DataPreprocessorInfo]


def classifiers() -> dict[str, ClassifierInfo]:
    """Get information about the classifiers available to auto-sklearn

    Returns
    -------
    dict[str, ClassifierInfo]
        The dict of classifiers and some info about them
    """
    return {
        name: ClassifierInfo(
            **{
                "type": cls,
                **_translate_properties(cls.get_properties(), "classifier"),
            }
        )
        for name, cls in ClassifierChoice.get_components().items()
    }


def regressors() -> dict[str, RegressorInfo]:
    """Get information about the regressors available to auto-sklearn

    Returns
    -------
    dict[str, RegressorInfo]
        The dict of regressors and some info about them
    """
    return {
        name: RegressorInfo(
            **{"type": cls, **_translate_properties(cls.get_properties(), "regressor")},
        )
        for name, cls in RegressorChoice.get_components().items()
    }


def feature_preprocessors() -> dict[str, FeaturePreprocessorInfo]:
    """Get information about the feature preprocessors available to auto-sklearn

    Returns
    -------
    dict[str, FeaturePreprocessorInfo]
        The dict of feature preprocessors and some info about them
    """
    return {
        name: FeaturePreprocessorInfo(
            **{
                "type": cls,
                **_translate_properties(cls.get_properties(), "f_preprocessor"),
            }
        )
        for name, cls in FeaturePreprocessorChoice.get_components().items()
    }


def data_preprocessors() -> dict[str, DataPreprocessorInfo]:
    """Get information about the data preprocessors available to auto-sklearn

    Returns
    -------
    dict[str, DataPreprocessorInfo]
        The dict of data preprocessors and some info about them
    """
    return {
        name: DataPreprocessorInfo(
            **{
                "type": cls,
                **_translate_properties(cls.get_properties(), "d_preprocessor"),
            }
        )
        for name, cls in DataPreprocessorChoice.get_components().items()
    }


def components() -> ComponentsInfo:
    """Get information about all of the components available to auto-sklearn

    Returns
    -------
    ComponentsInfo
        A dataclass with the items
        * classifiers
        * regressors
        * feature_preprocessors
        * data_preprocessors
    """
    return ComponentsInfo(
        classifiers=classifiers(),
        regressors=regressors(),
        feature_preprocessors=feature_preprocessors(),
        data_preprocessors=data_preprocessors(),
    )
