# -*- encoding: utf-8 -*-

__all__ = [
    'SimpleDataManager',
]

import numpy as np
import scipy.sparse
from ParamSklearn.implementations.OneHotEncoder import OneHotEncoder

from autosklearn.util import predict_RAM_usage


class SimpleDataManager(object):

    def __init__(self):
        self._data = dict()
        self._info = dict()

        self._basename = None
        self._feat_type = None
        self._encoder = None

    @property
    def basename(self):
        return self._basename

    @property
    def data(self):
        return self._data

    @property
    def info(self):
        return self._info

    @info.setter
    def info(self, value):
        self._info = value

    @property
    def feat_type(self):
        return self._feat_type

    @feat_type.setter
    def feat_type(self, value):
        self._feat_type = value

    @property
    def encoder(self):
        return self._encoder

    @encoder.setter
    def encoder(self, value):
        self._encoder = value

    def perform_hot_encoding(self):
        if not hasattr(self, '_data') and self._data is not None:
            raise ValueError('perform1HotEncoding can only be called when '
                             'data is loaded')
        if hasattr(self, '_encoder') and self._encoder is not None:
            raise ValueError('perform1HotEncoding can only be called on '
                             'non-encoded data.')

        sparse = True if self.info['is_sparse'] == 1 else False
        has_missing = True if self.info['has_missing'] else False

        to_encode = ['categorical']
        if has_missing:
            to_encode += ['binary']
        encoding_mask = [feat_type.lower() in to_encode
                         for feat_type in self._feat_type]

        categorical = [True if feat_type.lower() == 'categorical' else False
                       for feat_type in self._feat_type]

        predicted_RAM_usage = float(
            predict_RAM_usage(self.data['X_train'], categorical)) / pow(1024, 2)

        if predicted_RAM_usage > 1000:
            sparse = True

        if any(encoding_mask):
            encoder = OneHotEncoder(categorical_features=encoding_mask,
                                    dtype=np.float32,
                                    sparse=sparse)

            to_dence_flg = False
            for x in ['X_train', 'X_valid', 'X_test']:
                if x in self.data:
                    self.data[x] = encoder.fit_transform(self.data[x])
                    if x == 'X_train':
                        to_dence_flg = not sparse and scipy.sparse.issparse(self.data[x])
                    if to_dence_flg:
                        self.data[x] = self.data[x].todense()

            self._encoder = encoder
            self.info['is_sparse'] = 1 if sparse else 0

    def __repr__(self):
        return 'DataManager : ' + self._basename

    def __str__(self):
        info_text = '\n'.join(
            ['\t%s =  %s' % (x, str(self.info[x])) for x in self.info])

        data_text = ''

        for subset in self.data:
            dst = self.data[subset]  # data sub set
            data_text += '\t%s = %s %s %s\n' % (subset, type(dst),
                                          str(dst.shape),
                                          str(dst.dtype))
            if isinstance(dst, scipy.sparse.spmatrix):
                density = float(len(dst.data)) / dst.shape[0] / dst.shape[1]
                data_text += '\tdensity: %f\n' % density

        if isinstance(self._feat_type, np.ndarray):
            feat_type_str = str(self._feat_type.shape)
        elif isinstance(self._feat_type, list):
            feat_type_str = str(self._feat_type)
        else:
            raise ValueError("Feat type is't np.ndarray and list")

        return "DataManager : " \
               "{0}" \
               "Info: " \
               "{1}" \
               "Data:" \
               "{2}" \
               "Feat_type: " \
               "{3}".format(
            self._basename,
            info_text,
            data_text,
            feat_type_str,
        )
