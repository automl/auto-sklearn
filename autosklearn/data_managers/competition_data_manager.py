# -*- encoding: utf-8 -*-

# Functions performing various input/output operations for the ChaLearn
# AutoML challenge

# Main contributor: Arthur Pesah, August 2014
# Edits: Isabelle Guyon, October 2014

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.
from __future__ import print_function


__all__ = [
    'CompetitionDataManager',
]

import os
import re
import time

import numpy as np
import scipy.sparse
from autosklearn.constants import MULTILABEL_CLASSIFICATION, \
    STRING_TO_TASK_TYPES, MULTICLASS_CLASSIFICATION
from autosklearn.c_utils import read_dense_file, \
    read_sparse_file, read_sparse_binary_file, read_dense_file_unknown_width
from autosklearn.data_managers import SimpleDataManager
from autosklearn.util import convert_to_num, read_first_line, file_to_array

try:
    from autosklearn.data_managers import \
        read_first_line as c_read_first_line
    competition_c_functions_is_there = True
except Exception:
    competition_c_functions_is_there = False


def get_info_from_file(filename):
    """
    Get all information {attribute = value} pairs from the public.info file
    :param filename:
    :return:
    """
    info = {}
    with open(filename, 'r') as info_file:
        lines = info_file.readlines()
        features_list = list(
            map(lambda x: tuple(x.strip("\'").split(' = ')), lines)
        )

        for (key, value) in features_list:
            val = value.rstrip().strip("'").strip(' ')
            info[key] = int(val) if val.isdigit() else val

    return info


def get_format_data(info, filename):
    """
    Get the data format directly from the data file (in case we do not
    have an info file)
    :param info:
    :param filename:
    :return:
    """

    def read_data(filepath):
        if competition_c_functions_is_there:
            result = read_first_line(filepath)
        else:
            result = c_read_first_line(filepath)
        return result

    main_key = 'format'
    is_space = 'is_sparse'
    if main_key in info.keys():
        return info

    if is_space in info.keys():
        if info[is_space] == 0:
            info[main_key] = 'dense'
        else:
            info[main_key] = 'sparse' if ':' in read_data(filename)[
                0] else 'sparse_binary'
    else:
        data = read_data(filename)
        if ':' in data[0][0]:
            info[is_space] = 1
            info[main_key] = 'sparse'
        else:
            nbr_columns = len(data[0])
            for row in range(len(data)):
                if len(data[row]) != nbr_columns:
                    info[main_key] = 'sparse_binary'
            if main_key not in info.keys():
                info[main_key] = 'dense'
                info[is_space] = 0
    return info


class CompetitionDataManager(SimpleDataManager):
    """ This class aims at loading and saving data easily with a cache and at
    generating a dictionary (self.info) in which each key is a feature
    (e.g. : name, format, feat_num,...).

    Methods defined here are :
    __init__ (...)
        x.__init__([(feature, value)]) -> void
        Initialize the info dictionary with the tuples (feature, value)
        given as argument. It recognizes the type of value (int, string) and
        assign value to info[feature]. An unlimited number of tuple can be sent.

    getInfo (...)
        x.getInfo (filename) -> void
        Fill the dictionary with an info file. Each line of the info file must
        have this format 'feature' : value
        The information is obtained from the public.info file if it exists, or
        inferred from the data files

    getInfoFromFile (...)
        x.getInfoFromFile (filename) -> void
        Fill the dictionary with an info file. Each line of the info file must
        have this format 'feature' : value
    """

    def __init__(self, basename, input_dir, verbose=False, encode_labels=True):
        super(CompetitionDataManager, self).__init__()

        self._basename = basename
        if basename in input_dir:
            self.input_dir = input_dir
        else:
            self.input_dir = input_dir + '/' + basename + '/'

        info_file = os.path.join(self.input_dir, basename + '_public.info')
        self.get_info(info_file)
        self._feat_type = self.load_type(os.path.join(self.input_dir,
                                                      basename + '_feat.type'),
                                         verbose=verbose)

        Xtr = self.load_data(
            os.path.join(self.input_dir, basename + '_train.data'),
            self.info['train_num'], verbose=verbose)
        Ytr = self.load_label(
            os.path.join(self.input_dir, basename + '_train.solution'),
            self.info['train_num'], verbose=verbose)
        Xva = self.load_data(
            os.path.join(self.input_dir, basename + '_valid.data'),
            self.info['valid_num'], verbose=verbose)
        Xte = self.load_data(
            os.path.join(self.input_dir, basename + '_test.data'),
            self.info['test_num'], verbose=verbose)

        self._data['X_train'] = Xtr
        self._data['Y_train'] = Ytr
        self._data['X_valid'] = Xva
        self._data['X_test'] = Xte

        p = os.path.join(self.input_dir, basename + '_valid.solution')
        if os.path.exists(p):
            try:
                self._data['Y_valid'] = self.load_label(p,
                                                        self.info['valid_num'],
                                                        verbose=verbose)
            except (IOError, OSError):
                pass

        p = os.path.join(self.input_dir, basename + '_test.solution')
        if os.path.exists(p):
            try:
                self.data['Y_test'] = self.load_label(p, self.info['test_num'],
                                                      verbose=verbose)
            except (IOError, OSError) as e:
                pass

        if encode_labels:
            self.perform_hot_encoding()

    def load_data(self, filename, num_points, verbose=True):
        """ Get the data from a text file in one of 3 formats: matrix,
        sparse, binary_sparse"""
        if verbose:
            print('========= Reading ' + filename)
        start = time.time()

        if 'format' not in self.info:
            self.info.update(**get_format_data(self.info, filename))
        if competition_c_functions_is_there:
            data_func = {
                'dense': read_dense_file,
                'sparse': read_sparse_file,
                'sparse_binary': read_sparse_binary_file
            }

            data = data_func[self.info['format']](filename, num_points,
                                                  self.info['feat_num'])

            if scipy.sparse.issparse(data):
                if not np.all(data.indices >= 0):
                    raise ValueError('Sparse data must be 1-indexed, '
                                     'not 0-indexed.')
        else:
            data_func = {
                'dense': _data_dense,
                'sparse': _data_sparse,
                'sparse_binary': _data_binary_sparse
            }

            data = data_func[self.info['format']](filename, self._feat_type)

        end = time.time()
        if verbose:
            print('[+] Success in %5.2f sec' % (end - start))
        return data

    def load_label(self, filename, num_points, verbose=True):
        """Get the solution/truth values."""
        if verbose:
            print('========= Reading ' + filename)
        start = time.time()

        # IG: Here change to accommodate the new multiclass label format
        if competition_c_functions_is_there:
            if self.info['task'] == MULTILABEL_CLASSIFICATION:
                # cast into ints
                label = (read_dense_file_unknown_width(
                    filename, num_points)).astype(np.int)
            elif self.info['task'] == MULTICLASS_CLASSIFICATION:
                label = read_dense_file_unknown_width(
                    filename, num_points)
                # read the class from the only non zero entry in each line!
                # should be ints right away
                label = np.where(label != 0)[1]
            else:
                label = read_dense_file_unknown_width(
                    filename, num_points)
        else:
            if self.info['task'] == MULTILABEL_CLASSIFICATION:
                label = self._data(filename)
            elif self.info['task'] == MULTICLASS_CLASSIFICATION:
                label = convert_to_num(self._data(filename))
            else:
                label = np.ravel(data_util.data(filename)
                                 )  # get a column vector

        end = time.time()
        if verbose:
            print('[+] Success in %5.2f sec' % (end - start))
        return label

    def load_type(self, filename, verbose=True):
        """Get the variable types."""
        if verbose:
            print('========= Reading ' + filename)
        start = time.time()
        type_list = []
        if os.path.isfile(filename):
            if competition_c_functions_is_there:
                type_list = file_to_array(
                    filename,
                    verbose=False)
            else:
                type_list = file_to_array(filename, verbose=False)
        else:
            n = self.info['feat_num']
            type_list = [self.info['feat_type']] * n
        type_list = np.array(type_list).ravel()
        end = time.time()
        if verbose:
            print('[+] Success in %5.2f sec' % (end - start))
        return type_list

    def get_info(self, filename, verbose=True):
        """ Get all information {attribute = value} pairs from the filename (public.info file),
              if it exists, otherwise, output default values"""
        if filename is None:
            basename = self._basename
            input_dir = self.input_dir
        else:
            # Split away the _public.info (anyway, I don't know why its
            # there... the dataset name is known from the call)
            basename = '_'.join(os.path.basename(filename).split('_')[:-1])
            input_dir = os.path.dirname(filename)
        if os.path.exists(filename):
            self.info.update(**get_info_from_file(filename))
            print('Info file found : ' + os.path.abspath(filename))
            # Finds the data format ('dense', 'sparse', or 'sparse_binary')
            self.info.update(
                **get_format_data(self.info, os.path.join(input_dir,
                                                          basename + '_train.data')))
        else:
            raise NotImplementedError('The user must always provide an info '
                                      'file.')

        self.info['task'] = STRING_TO_TASK_TYPES[self.info['task']]

        return self.info


def _data_dense(filename, feat_type=None, verbose=False):
    # The 2nd parameter makes possible a using of the 3 functions of data
    # reading (data, data_sparse, data_binary_sparse) without changing
    # parameters

    # This code is based on scipy.io.arff.arff_load
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
            # yield tuple([np.float64(row[i]) for i in elems])
            yield tuple([row[i] for i in elems])

    with open(filename) as fh:
        a = generator(fh, delim=' ')
        # No error should happen here: it is a bug otherwise
        data = np.fromiter(a, descr)

        data = data.view(np.float32).reshape((len(data), -1))
        return data


def _data_sparse(filename, feat_type):
    # This function takes as argument a file representing a sparse matrix
    # sparse_matrix[i][j] = "a:b" means matrix[i][a] = b
    # It converts it into a numpy array, using sparse_list_to_array function,
    # and returns this array
    sparse_list = _sparse_file_to_sparse_list(filename)
    return _sparse_list_to_csr_sparse(sparse_list, len(feat_type))


def _data_binary_sparse(filename, feat_type):
    # This function takes as an argument a file representing a binary sparse
    # matrix
    # binary_sparse_matrix[i][j] = a means matrix[i][j] = 1
    # It converts it into a numpy array an returns this array.

    inner_data = file_to_array(filename)
    nbr_samples = len(inner_data)
    # the construction is easier w/ dok_sparse
    dok_sparse = scipy.sparse.dok_matrix((nbr_samples, len(feat_type)))
    print('Converting {} to dok sparse matrix'.format(filename))
    for row in range(nbr_samples):
        for feature in inner_data[row]:
            dok_sparse[row, int(feature) - 1] = 1
    print('Converting {} to csr sparse matrix'.format(filename))
    return dok_sparse.tocsr()


def _sparse_file_to_sparse_list(filename, verbose=True):
    # Converts a sparse data file to a sparse list, so that:
    # sparse_list[i][j] = (a,b) means matrix[i][a]=b
    data_file = open(filename, 'r')
    if verbose:
        print('Reading {}...'.format(filename))
    lines = data_file.readlines()
    if verbose:
        print('Converting {} to correct array')
    data = [lines[i].split(' ') for i in range(len(lines))]
    if verbose:
        print('Converting {} to sparse list'.format(filename))

    _converter = lambda a_: (int(a_[0]), np.float32(float(a_[1])))
    return [[_converter(data[i][j].rstrip().split(':'))
             for j in range(len(data[i])) if data[i][j] != '\n']
            for i in range(len(data))]


def _sparse_list_to_csr_sparse(sparse_list, nbr_features, verbose=True):
    # This function takes as argument a matrix of tuple representing a sparse
    # matrix and the number of features.
    # sparse_list[i][j] = (a,b) means matrix[i][a]=b
    # It converts it into a scipy csr sparse matrix
    nbr_samples = len(sparse_list)
    # construction easier w/ dok_sparse...
    dok_sparse = scipy.sparse.dok_matrix((nbr_samples, nbr_features),
                                         dtype=np.float32)
    if verbose:
        print('\tConverting sparse list to dok sparse matrix')
    for row in range(nbr_samples):
        for column in range(len(sparse_list[row])):
            (feature, value) = sparse_list[row][column]
            dok_sparse[row, feature - 1] = value
    if verbose:
        print('\tConverting dok sparse matrix to csr sparse matrix')
        # but csr better for shuffling data or other tricks
    return dok_sparse.tocsr()
