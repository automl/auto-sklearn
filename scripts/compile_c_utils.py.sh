#!/usr/bin/env bash
# WARNING: hindi code

# Cython library installed in venv
VENV_FOLDER="/home/warmonger/Develop/venv/autosk2"

SCRIPTS_FOLDER=$(dirname "$0")
cd ..
BASE_FOLDER=$(pwd)
C_UTILS_FOLDER=$BASE_FOLDER/autosklearn/c_utils
cd $C_UTILS_FOLDER


source $VENV_FOLDER/bin/activate

python setup.py build_ext --build-lib $BASE_FOLDER --build-temp /tmp/c_utils


deactivate