import numpy as np
import scipy.sparse

from ParamSklearn.implementations.OneHotEncoder import OneHotEncoder

from autosklearn.data import util as data_util


class DataManager(object):
    def __init__(self):
        self._data = dict()
        self._info = dict()

    @property
    def data(self):
        return self._data

    @property
    def info(self):
        return self._info

    def perform1HotEncoding(self):
        if not hasattr(self, "data"):
            raise ValueError("perform1HotEncoding can only be called when "
                             "data is loaded")
        if hasattr(self, "encoder_"):
            raise ValueError("perform1HotEncoding can only be called on "
                             "non-encoded data.")
        self.encoder_ = None

        sparse = True if self.info['is_sparse'] == 1 else False
        has_missing = True if self.info['has_missing'] else False

        to_encode = ['categorical']
        if has_missing:
            to_encode += ['binary']
        encoding_mask = [feat_type.lower() in to_encode
                         for feat_type in self.feat_type]

        categorical = [True if feat_type.lower() == 'categorical' else False
                       for feat_type in self.feat_type]

        predicted_RAM_usage = float(
            data_util.predict_RAM_usage(
                self.data['X_train'], categorical)) / 1024 / 1024

        if predicted_RAM_usage > 1000:
            sparse = True

        if any(encoding_mask):
            encoder = OneHotEncoder(categorical_features=encoding_mask,
                                    dtype=np.float32, sparse=sparse)
            self.data['X_train'] = encoder.fit_transform(self.data['X_train'])
            if 'X_valid' in self.data:
                self.data['X_valid'] = encoder.transform(self.data['X_valid'])
            if 'X_test' in self.data:
                self.data['X_test'] = encoder.transform(self.data['X_test'])

            if not sparse and scipy.sparse.issparse(self.data['X_train']):
                self.data['X_train'] = self.data['X_train'].todense()
                if 'X_valid' in self.data:
                    self.data['X_valid'] = self.data['X_valid'].todense()
                if 'X_test' in self.data:
                    self.data['X_test'] = self.data['X_test'].todense()

            self.encoder_ = encoder
            self.info['is_sparse'] = 1 if sparse else 0

    def __repr__(self):
        return "DataManager : " + self.basename

    def __str__(self):
        val = "DataManager : " + self.basename + "\ninfo:\n"
        for item in self.info:
            val = val + "\t" + item + " = " + str(self.info[item]) + "\n"
        val = val + "data:\n"

        for subset in self.data:
            val = val + "\t%s = %s %s %s\n" % (subset, type(self.data[subset]), \
                                               str(self.data[subset].shape),
                                               str(self.data[subset].dtype))
            if isinstance(self.data[subset], scipy.sparse.spmatrix):
                val = val + "\tdensity: %f\n" % \
                            (float(len(self.data[subset].data)) /
                             self.data[subset].shape[0] /
                             self.data[subset].shape[1])
        val = val + "feat_type:\tarray" + str(self.feat_type.shape) + "\n"
        return val