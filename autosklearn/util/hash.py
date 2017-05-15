import hashlib

import scipy.sparse


def hash_array_or_matrix(X):
    m = hashlib.md5()

    if scipy.sparse.issparse(X):
        m.update(X.indices)
        m.update(X.indptr)
        m.update(X.data)
        m.update(str(X.shape).encode('utf8'))
    else:
        if X.flags['C_CONTIGUOUS']:
            m.update(X.data)
            m.update(str(X.shape).encode('utf8'))
        else:
            m.update(X.T.data)
            m.update(str(X.T.shape).encode('utf8'))

    hash = m.hexdigest()
    return hash