# -*- encoding: utf-8 -*-

# Functions performing various input/output operations for the ChaLearn
# AutoML challenge

from __future__ import print_function
import os
import re
import time

import numpy as np
import scipy.sparse

from autosklearn.constants import MULTILABEL_CLASSIFICATION, \
    STRING_TO_TASK_TYPES,  MULTICLASS_CLASSIFICATION, STRING_TO_METRIC
from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.util import convert_to_num


import autosklearn.data.competition_c_functions as competition_c_functions


def load_labels(filename):
    return np.genfromtxt(filename, dtype=np.float64)


class CompetitionDataManager(AbstractDataManager):

    def __init__(self, name, encode_labels=True, max_memory_in_mb=1048576):
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
            max_memory_in_mb = max_memory_in_mb)
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

        if encode_labels:
            self.perform1HotEncoding()

    def load_data(self, filename, num_points, max_memory_in_mb):
        """Get the data from a text file in one of 3 formats:
        matrix, sparse, binary_sparse"""
        # if verbose:
        #     print('========= Reading ' + filename)
        # start = time.time()

        if 'format' not in self.info:
            self.get_format_data(filename)

        data_func = {
            'dense': competition_c_functions.read_dense_file,
            'sparse': competition_c_functions.read_sparse_file,
            'sparse_binary': competition_c_functions.read_sparse_binary_file
        }

        data = data_func[self.info['format']](filename, num_points,
                                              self.info['feat_num'], max_memory_in_mb)

        if scipy.sparse.issparse(data):
            if not np.all(data.indices >= 0):
                raise ValueError('Sparse data must be 1-indexed, '
                                 'not 0-indexed.')

        # end = time.time()
        # if verbose:
        #     print('[+] Success in %5.2f sec' % (end - start))
        return data

    def load_label(self, filename, num_points):
        """Get the solution/truth values."""
        # if verbose:
        #     print('========= Reading ' + filename)
        # start = time.time()

        # IG: Here change to accommodate the new multiclass label format
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

        # end = time.time()
        # if verbose:
        #     print('[+] Success in %5.2f sec' % (end - start))
        return label

    def load_type(self, filename):
        """Get the variable types."""
        # if verbose:
        #     print('========= Reading ' + filename)
        # start = time.time()
        # type_list = []
        if os.path.isfile(filename):
            type_list = competition_c_functions.file_to_array(filename)
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
        self.info['metric'] = STRING_TO_METRIC[self.info['metric']]

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
                data = competition_c_functions.read_first_line(filename)
                if ':' in data[0]:
                    self.info['format'] = 'sparse'
                else:
                    self.info['format'] = 'sparse_binary'
        else:
            data = competition_c_functions.file_to_array(filename)
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
