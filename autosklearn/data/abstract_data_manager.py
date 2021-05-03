import abc
from typing import Any, Dict, List

import numpy as np

import scipy.sparse

from autosklearn.pipeline.components.data_preprocessing.data_preprocessing \
    import DataPreprocessor


class AbstractDataManager():
    __metaclass__ = abc.ABCMeta

    def __init__(self, name: str):

        self._data = dict()  # type: Dict
        self._info = dict()  # type: Dict
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> Dict[str, np.ndarray]:
        return self._data

    @property
    def info(self) -> Dict[str, Any]:
        return self._info

    @property
    def feat_type(self) -> List[str]:
        return self._feat_type

    @feat_type.setter
    def feat_type(self, value: List[str]) -> None:
        self._feat_type = value

    @property
    def encoder(self) -> DataPreprocessor:
        return self._encoder

    @encoder.setter
    def encoder(self, value: DataPreprocessor) -> DataPreprocessor:
        self._encoder = value

    def __repr__(self) -> str:
        return 'DataManager : ' + self.name

    def __str__(self) -> str:
        val = 'DataManager : ' + self.name + '\ninfo:\n'
        for item in self.info:
            val = val + '\t' + item + ' = ' + str(self.info[item]) + '\n'
        val = val + 'data:\n'

        for subset in self.data:
            val = val + '\t%s = %s %s %s\n' % (subset, type(self.data[subset]),
                                               str(self.data[subset].shape),
                                               str(self.data[subset].dtype))
            if isinstance(self.data[subset], scipy.sparse.spmatrix):
                val = val + '\tdensity: %f\n' % \
                            (float(len(self.data[subset].data)) /
                             self.data[subset].shape[0] /
                             self.data[subset].shape[1])
        val = val + 'feat_type:\t' + str(self.feat_type) + '\n'
        return val
