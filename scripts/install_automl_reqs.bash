#!/bin/bash


echo "Install HPOlibConfigSpace"
pip install git+https://github.com/mfeurer/HPOlibConfigSpace#egg=HPOlibConfigSpace0.1dev

echo "Install ParamSklearn"
pip install git+https://git@bitbucket.org/mfeurer/paramsklearn.git@73d8643b2849db753ddc7b8909d01e6cee9bafc6#egg=ParamSklearn --no-deps

echo "Install HPOlib"
pip install git+https://github.com/automl/HPOlib@development#egg=HPOlib

echo "Install pyMetaLearn"
pip install git+https://bitbucket.org/mfeurer/pymetalearn/#egg=pyMetaLearn

