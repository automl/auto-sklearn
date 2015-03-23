import re

import numpy as np


def convert_file_to_array(filename, feat_type):
    r_comment = re.compile(r'^%')
    # Match an empty line
    r_empty = re.compile(r'^\s+$')
    descr = [(str(i), np.float32) for i in range(len(feat_type))]

    def generator(row_iter, delim=','):
        # Copied from scipy.io.arff.arffread
        raw = next(row_iter)
        while r_empty.match(raw) or r_comment.match(raw):
            raw = next(row_iter)

        # 'compiling' the range since it does not change
        # Note, I have already tried zipping the converters and
        # row elements and got slightly worse performance.
        elems = list(range(len(feat_type)))

        row = raw.split(delim)
        # yield tuple([np.float64(row[i]) for i in elems])
        yield tuple([row[i] for i in elems])
        for raw in row_iter:
            while r_comment.match(raw) or r_empty.match(raw):
                raw = next(row_iter)
            row = raw.split(delim)
            #yield tuple([np.float64(row[i]) for i in elems])
            yield tuple([row[i] for i in elems])

    with open(filename) as fh:
        a = generator(fh, delim=" ")
        # No error should happen here: it is a bug otherwise
        data = np.fromiter(a, descr)

    data = data.view(np.float32).reshape((len(data), -1))
    return data
