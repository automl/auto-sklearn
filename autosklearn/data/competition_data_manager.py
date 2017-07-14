# -*- encoding: utf-8 -*-

# Functions performing various input/output operations for the ChaLearn
# AutoML challenge

import os
import re
import warnings

import numpy as np
import scipy.sparse

from autosklearn.constants import MULTILABEL_CLASSIFICATION, \
    STRING_TO_TASK_TYPES, MULTICLASS_CLASSIFICATION
from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.util import convert_to_num
try:
    import autosklearn.data.competition_c_functions as competition_c_functions

    competition_c_functions_is_there = True
except Exception:
    competition_c_functions_is_there = False


def data_dense(filename, feat_type=None):
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


def data_sparse(filename, feat_type):
    # This function takes as argument a file representing a sparse matrix
    # sparse_matrix[i][j] = "a:b" means matrix[i][a] = b
    # It converts it into a numpy array, using sparse_list_to_array function,
    # and returns this array
    sparse_list = sparse_file_to_sparse_list(filename)
    return sparse_list_to_csr_sparse(sparse_list, len(feat_type))


def data_binary_sparse(filename, feat_type):
    # This function takes as an argument a file representing a binary sparse
    # matrix
    # binary_sparse_matrix[i][j] = a means matrix[i][j] = 1
    # It converts it into a numpy array an returns this array.

    inner_data = file_to_array(filename)
    nbr_samples = len(inner_data)
    # the construction is easier w/ dok_sparse
    dok_sparse = scipy.sparse.dok_matrix((nbr_samples, len(feat_type)))
    # print('Converting {} to dok sparse matrix'.format(filename))
    for row in range(nbr_samples):
        for feature in inner_data[row]:
            dok_sparse[row, int(feature) - 1] = 1
    # print('Converting {} to csr sparse matrix'.format(filename))
    return dok_sparse.tocsr()


def file_to_array(filename):
    # Converts a file to a list of list of STRING; It differs from
    # np.genfromtxt in that the number of columns doesn't need to be constant
    with open(filename, 'r') as data_file:
        # if verbose:
        # print('Reading {}...'.format(filename))
        lines = data_file.readlines()
        # if verbose:
        #     print('Converting {} to correct array...'.format(filename))
        data = [lines[i].strip().split() for i in range(len(lines))]
    return data


def read_first_line(filename):
    # Read fist line of file
    with open(filename, 'r') as data_file:
        line = data_file.readline()
        data = line.strip().split()
    return data


def sparse_file_to_sparse_list(filename):
    # Converts a sparse data file to a sparse list, so that:
    # sparse_list[i][j] = (a,b) means matrix[i][a]=b
    data_file = open(filename, 'r')
    # if verbose:
    # print('Reading {}...'.format(filename))
    lines = data_file.readlines()
    # if verbose:
    #     print('Converting {} to correct array')
    data = [lines[i].split(' ') for i in range(len(lines))]
    # if verbose:
    #     print('Converting {} to sparse list'.format(filename))

    _converter = lambda a_: (int(a_[0]), np.float32(float(a_[1])))
    return [[_converter(data[i][j].rstrip().split(':'))
             for j in range(len(data[i])) if data[i][j] != '\n']
            for i in range(len(data))]


def sparse_list_to_csr_sparse(sparse_list, nbr_features):
    # This function takes as argument a matrix of tuple representing a sparse
    # matrix and the number of features.
    # sparse_list[i][j] = (a,b) means matrix[i][a]=b
    # It converts it into a scipy csr sparse matrix
    nbr_samples = len(sparse_list)
    # construction easier w/ dok_sparse...
    dok_sparse = scipy.sparse.dok_matrix((nbr_samples, nbr_features),
                                         dtype=np.float32)
    # if verbose:
    # print('\tConverting sparse list to dok sparse matrix')
    for row in range(nbr_samples):
        for column in range(len(sparse_list[row])):
            (feature, value) = sparse_list[row][column]
            dok_sparse[row, feature - 1] = value
    # if verbose:
    #    print('\tConverting dok sparse matrix to csr sparse matrix')
    #     # but csr better for shuffling data or other tricks
    return dok_sparse.tocsr()


def load_labels(filename):
    return np.genfromtxt(filename, dtype=np.float64)


class CompetitionDataManager(AbstractDataManager):
    def __init__(self, name, max_memory_in_mb=1048576):
        """ max_memory_size in Mb """
        if name.endswith("/"):
            name = name[:-1]
        input_dir = os.path.dirname(name)
        if not input_dir:
            input_dir = "."
        name = os.path.basename(name)

        super(CompetitionDataManager, self).__init__(name)
        self.input_dir = os.path.join(input_dir, name)

        info_file = os.path.join(self.input_dir, self.name + '_public.info')
        self.get_info(info_file)
        self.feat_type = self.load_type(os.path.join(self.input_dir,
                                                     self.name + '_feat.type'))

        # apply memory limit here for really large training sets
        Xtr = self.load_data(
            os.path.join(self.input_dir, self.name + '_train.data'),
            self.info['train_num'],
            max_memory_in_mb=max_memory_in_mb)
        Ytr = self.load_label(
            os.path.join(self.input_dir, self.name + '_train.solution'),
            self.info['train_num'])
        # no restriction here
        Xva = self.load_data(
            os.path.join(self.input_dir, self.name + '_valid.data'),
            self.info['valid_num'],
            max_memory_in_mb=1048576)
        Xte = self.load_data(
            os.path.join(self.input_dir, self.name + '_test.data'),
            self.info['test_num'],
            max_memory_in_mb=1048576)

        # update the info in case the data has been cut off
        self.info['train_num'] = Xtr.shape[0]

        self._data['X_train'] = Xtr
        self._data['Y_train'] = Ytr
        self._data['X_valid'] = Xva
        self._data['X_test'] = Xte

        p = os.path.join(self.input_dir, self.name + '_valid.solution')
        if os.path.exists(p):
            try:
                self._data['Y_valid'] = self.load_label(p,
                                                        self.info['valid_num'])
            except (IOError, OSError):
                pass

        p = os.path.join(self.input_dir, self.name + '_test.solution')
        if os.path.exists(p):
            try:
                self.data['Y_test'] = self.load_label(p, self.info['test_num'])
            except (IOError, OSError):
                pass

    def load_data(self, filename, num_points, max_memory_in_mb):
        """Get the data from a text file in one of 3 formats:
        matrix, sparse, binary_sparse"""
        # if verbose:
        # print('========= Reading ' + filename)
        # start = time.time()

        if 'format' not in self.info:
            self.get_format_data(filename)
        if competition_c_functions_is_there:
            data_func = {
                'dense': competition_c_functions.read_dense_file,
                'sparse': competition_c_functions.read_sparse_file,
                'sparse_binary': competition_c_functions.read_sparse_binary_file
            }

            data = data_func[self.info['format']](filename, num_points,
                                                  self.info['feat_num'],
                                                  max_memory_in_mb)

            if scipy.sparse.issparse(data):
                if not np.all(data.indices >= 0):
                    raise ValueError('Sparse data must be 1-indexed, '
                                     'not 0-indexed.')
        else:
            data_func = {
                'dense': data_dense,
                'sparse': data_sparse,
                'sparse_binary': data_binary_sparse
            }

            data = data_func[self.info['format']](filename, self.feat_type)

        # end = time.time()
        # if verbose:
        #     print('[+] Success in %5.2f sec' % (end - start))
        return data

    def load_label(self, filename, num_points):
        """Get the solution/truth values."""
        # if verbose:
        # print('========= Reading ' + filename)
        # start = time.time()

        # IG: Here change to accommodate the new multiclass label format
        if competition_c_functions_is_there:
            if self.info['task'] == MULTILABEL_CLASSIFICATION:
                # cast into ints
                label = (competition_c_functions.read_dense_file_unknown_width(
                    filename, num_points)).astype(np.int)
            elif self.info['task'] == MULTICLASS_CLASSIFICATION:
                label = competition_c_functions.read_dense_file_unknown_width(
                    filename, num_points)
                # read the class from the only non zero entry in each line!
                # should be ints right away
                label = np.where(label != 0)[1]
            else:
                label = competition_c_functions.read_dense_file_unknown_width(
                    filename, num_points)
        else:
            if self.info['task'] == MULTILABEL_CLASSIFICATION:
                label = load_labels(filename)
            elif self.info['task'] == MULTICLASS_CLASSIFICATION:
                label = convert_to_num(load_labels(filename))
            else:
                label = np.ravel(load_labels(filename))  # get a column vector

        # end = time.time()
        # if verbose:
        #     print('[+] Success in %5.2f sec' % (end - start))
        return label

    def load_type(self, filename):
        """Get the variable types."""
        # if verbose:
        # print('========= Reading ' + filename)
        # start = time.time()
        # type_list = []
        if os.path.isfile(filename):
            if competition_c_functions_is_there:
                type_list = competition_c_functions.file_to_array(filename)
            else:
                type_list = file_to_array(filename)
        else:
            n = self.info['feat_num']
            type_list = [self.info['feat_type']] * n
        type_list = np.array(type_list).ravel()
        # end = time.time()
        # if verbose:
        #     print('[+] Success in %5.2f sec' % (end - start))
        return type_list

    def get_info(self, filename):
        """Get all information {key = value} pairs from the filename
        (public.info file), if it exists, otherwise, output default values"""
        if filename is None:
            basename = self.name
            input_dir = self.input_dir
        else:
            # Split away the _public.info (anyway, I don't know why its
            # there... the dataset name is known from the call)
            basename = '_'.join(os.path.basename(filename).split('_')[:-1])
            input_dir = os.path.dirname(filename)
        if os.path.exists(filename):
            self.get_info_from_file(filename)
            # print('Info file found : ' + os.path.abspath(filename))
            # Finds the data format ('dense', 'sparse', or 'sparse_binary')
            self.get_format_data(os.path.join(input_dir,
                                              basename + '_train.data'))
        else:
            raise NotImplementedError('The user must always provide an info '
                                      'file.')

        self.info['task'] = STRING_TO_TASK_TYPES[self.info['task']]
        warnings.warn('auto-sklearn will no longer take the metric given by '
                      'the data manager into account. Please specify the '
                      'metric when calling fit().')

        return self.info

    def get_info_from_file(self, filename):
        """Get all information {key = value} pairs from public.info file"""
        with open(filename, 'r') as info_file:
            lines = info_file.readlines()
            features_list = list(
                map(lambda x: tuple(x.strip("\'").split(' = ')), lines))
            for (key, value) in features_list:
                self.info[key] = value.rstrip().strip("'").strip(' ')
                if self.info[key].isdigit(
                ):  # if we have a number, we want it to be an integer
                    self.info[key] = int(self.info[key])
        return self.info

    def get_format_data(self, filename):
        """Get the data format directly from the data file (in case we do not
        have an info file)"""
        if 'format' in self.info.keys():
            return self.info['format']
        if 'is_sparse' in self.info.keys():
            if self.info['is_sparse'] == 0:
                self.info['format'] = 'dense'
            else:
                if competition_c_functions_is_there:
                    data = competition_c_functions.read_first_line(filename)
                else:
                    data = read_first_line(filename)
                if ':' in data[0]:
                    self.info['format'] = 'sparse'
                else:
                    self.info['format'] = 'sparse_binary'
        else:
            if competition_c_functions_is_there:
                data = competition_c_functions.file_to_array(filename)
            else:
                data = file_to_array(filename)
            if ':' in data[0][0]:
                self.info['is_sparse'] = 1
                self.info['format'] = 'sparse'
            else:
                nbr_columns = len(data[0])
                for row in range(len(data)):
                    if len(data[row]) != nbr_columns:
                        self.info['format'] = 'sparse_binary'
                if 'format' not in self.info.keys():
                    self.info['format'] = 'dense'
                    self.info['is_sparse'] = 0
        return self.info['format']
