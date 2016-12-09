import hashlib


def hash_numpy_array(X):
    m = hashlib.md5()

    if X.flags['C_CONTIGUOUS']:
        m.update(X.data)
    else:
        m.update(X.T.data)

    hash = m.hexdigest()
    return hash