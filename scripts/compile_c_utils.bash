#!/usr/bin/env bash
# WARNING: hindi code

BASE_FOLDER=$(pwd)
C_UTILS_FOLDER=$BASE_FOLDER/autosklearn/c_utils
cd $C_UTILS_FOLDER


python setup.py build_ext --build-lib $BASE_FOLDER --build-temp /tmp/c_utils