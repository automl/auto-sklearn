import abc
from typing import Dict

from sklearn.base import BaseEstimator

from autosklearn.automl_common.common.ensemble_building.abstract_ensemble import AbstractEnsemble\
    as CommonAbstractEnsemble
from autosklearn.pipeline.base import BasePipeline


class AbstractEnsemble(CommonAbstractEnsemble):

    @abc.abstractmethod
    def to_sklearn(self, models: Dict[str, BasePipeline]) -> BaseEstimator:
        raise NotImplementedError()
