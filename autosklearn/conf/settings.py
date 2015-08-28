# -*- encoding: utf-8 -*-
from os.path import dirname, join, normpath

BASE_DIR = normpath(join(dirname(__file__), '..', '..'))
LIB_DIR = join(BASE_DIR, 'autosklearn')

DOWNLOAD_DIRECTORY = join(BASE_DIR, '.downloads')
BINARIES_DIRECTORY = join(LIB_DIR, 'binaries')

RES_FOLDER = join(LIB_DIR, 'resources')
SCRIPT_FOLDER = join(LIB_DIR, 'scripts')
LIB_SCRIPT_FOLDER = join(BASE_DIR, 'scripts')


META_LEARNING_FOLDER = join(RES_FOLDER, 'meta_learning')
METADATA_DIRECTORY = META_LEARNING_FOLDER

SCRIPT_ENSEMBLE_SELECTION = join(SCRIPT_FOLDER, 'ensemble_selection_script.py')
SCRIPT_COMPILE_C_UTILS = join(LIB_SCRIPT_FOLDER, 'compile_c_utils.bash')
