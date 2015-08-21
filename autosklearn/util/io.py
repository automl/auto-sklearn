# -*- encoding: utf-8 -*-
import fnmatch
from glob import glob
import os
import subprocess

from conf.settings import BINARIES_DIRECTORY

def ls(filename):
    return sorted(glob(filename))

def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def check_pid(pid):
    """
    Check For the existence of a unix pid.
    :param pid:
    :return:
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def read_first_line(filename):
    # Read fist line of file
    data = []
    with open(filename, 'r') as data_file:
        line = data_file.readline()
        data = line.strip().split()
    return data

def file_to_array(filename, verbose=False):
    # Converts a file to a list of list of STRING; It differs from
    # np.genfromtxt in that the number of columns doesn't need to be constant
    data = []
    with open(filename, 'r') as data_file:
        if verbose:
            print('Reading {}...'.format(filename))
        lines = data_file.readlines()
        if verbose:
            print('Converting {} to correct array...'.format(filename))
        data = [lines[i].strip().split() for i in range(len(lines))]
    return data


def find_files(folder, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches


def _clojure_search_file_in_folder(default_folder):
    folder = default_folder

    def search_file(prog_name):
        try:
            files = find_files(folder, prog_name)
            assert files
            result = os.path.normpath(files[0])
        except Exception as e:
            result = None
        return result

    return search_file


search_prog_in_binaries = _clojure_search_file_in_folder(BINARIES_DIRECTORY)


def search_prog(prog_name):
    try:
        p = subprocess.Popen(["whereis", prog_name], stdout=subprocess.PIPE)
        (output, _) = p.communicate()
        assert output, "Not found %s" % prog_name
        paths = output.split()
        assert paths, "Not found (whereis) prog - %s" % prog_name
        result = paths[1]
    except Exception as e:
        result = None
    return result
