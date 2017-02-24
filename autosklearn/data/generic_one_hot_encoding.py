import numpy as np
import scipy.sparse

from autosklearn.pipeline.implementations.OneHotEncoder import OneHotEncoder

from autosklearn.util import predict_RAM_usage


def perform_one_hot_encoding(sparse, categorical, data):

    predicted_RAM_usage = float(predict_RAM_usage(data[0], categorical)) / 1024 / 1024

    if predicted_RAM_usage > 1000:
        sparse = True

    rvals = []
    if any(categorical):
        encoder = OneHotEncoder(categorical_features=categorical,
                                dtype=np.float32,
                                sparse=sparse)
        rvals.append(encoder.fit_transform(data[0]))
        for d in data[1:]:
            rvals.append(encoder.transform(d))

        if not sparse and scipy.sparse.issparse(rvals[0]):
            for i in range(len(rvals)):
                rvals[i] = rvals[i].todense()
    else:
        rvals = data

    return rvals, sparse