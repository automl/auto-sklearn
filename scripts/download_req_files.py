# -*- encoding: utf-8 -*-
from __future__ import print_function
try:
    from __init__ import *
except ImportError:
    pass

import os
import shutil
import subprocess
import sys
import tarfile
import urllib
from os.path import join

from conf.settings import DOWNLOAD_DIRECTORY, BINARIES_DIRECTORY

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

SMAC_DOWNLOAD_LOCATION = 'http://aad.informatik.uni-freiburg.de/~feurerm/'
SMAC_TAR_NAME = 'smac-v2.08.01-development-1.tar.gz'
METADATA_LOCATION = 'http://aad.informatik.uni-freiburg.de/~feurerm/'
METADATA_TAR_NAME = 'metadata_automl1_000.tar.gz'
RUNSOLVER_LOCATION = 'http://www.cril.univ-artois.fr/~roussel/runsolver/'
RUNSOLVER_TAR_NAME = 'runsolver-3.3.4.tar.bz2'


def load_file(data):
    download_url, filename = data
    # This can fail ungracefully, because having these files is
    # crucial to AutoSklearn!
    print('Download file: %s' % download_url)
    filepath = os.path.join(DOWNLOAD_DIRECTORY, filename)
    urllib.urlretrieve(os.path.join(download_url, filename), filename=filepath)
    return filepath


def extract_file(filename):
    print('Extract file: %s' % filename)
    tfile = tarfile.open(os.path.join(DOWNLOAD_DIRECTORY, filename))
    tfile.extractall(
        os.path.join(DOWNLOAD_DIRECTORY,
                     filename.replace('.tar.gz', '').replace('.tar.bz2', '')))


def clean_directory_dir(dirpath):
    try:
        shutil.rmtree(dirpath)
    except Exception:
        pass

    try:
        os.makedirs(dirpath)
    except Exception:
        pass


def copy_file(old_pos, new_pos):
    shutil.move(old_pos, new_pos)


def build_runsolver(folder_path):
    cur_pwd = os.getcwd()

    os.chdir(folder_path)
    subprocess.check_call('make')
    os.chdir(cur_pwd)


def run():
    map(clean_directory_dir, [DOWNLOAD_DIRECTORY, BINARIES_DIRECTORY])
    with open(os.path.join(BINARIES_DIRECTORY, '__init__.py'), 'w'):
        pass

    map(extract_file, map(load_file,
                          [(SMAC_DOWNLOAD_LOCATION, SMAC_TAR_NAME),
                           (RUNSOLVER_LOCATION, RUNSOLVER_TAR_NAME)]))

    sys.stdout.write('Building runsolver\n')
    runsolver_source_path = os.path.join(DOWNLOAD_DIRECTORY, 'runsolver-3.3.4',
                                         'runsolver', 'src')
    build_runsolver(runsolver_source_path)

    map(lambda x_y: shutil.move(x_y[0], x_y[1]), [
        (join(runsolver_source_path, 'runsolver'),
         join(BINARIES_DIRECTORY, 'runsolver')),
        (join(DOWNLOAD_DIRECTORY, SMAC_TAR_NAME.replace('.tar.gz', '')),
         BINARIES_DIRECTORY),
        # (os.path.join(DOWNLOAD_DIRECTORY,
        #               METADATA_TAR_NAME.replace(".tar.gz", ""),
        #               "files"),
        #  METADATA_DIRECTORY)
    ])

    map(shutil.rmtree, [
        DOWNLOAD_DIRECTORY,  # BINARIES_DIRECTORY,
        # METADATA_DIRECTORY
    ])


if __name__ == '__main__':
    run()
