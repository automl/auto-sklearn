### Download and build the documentation ###

    pip install scikit-learn==0.15.2
    pip install git+https://github.com/mfeurer/HPOlibConfigSpace#egg=HPOlibConfigSpace0.1dev
    git clone https://bitbucket.org/mfeurer/paramsklearn.git
    cd paramsklearn
    python setup.py install
    python setup.py test

Installation with `pip`

    pip install numpy scipy scikit-learn==0.15.2 numpydoc sphinx
    pip install git+https://github.com/mfeurer/HPOlibConfigSpace#egg=HPOlibConfigSpace0.1dev
    pip install --editable git+https://bitbucket.org/mfeurer/paramsklearn#egg=ParamSklearn

To build the documentation you also need the packages `sphinx` and `numpydoc`.

    pip install sphinx
    pip install numpydoc
    make html
    firefox `pwd`/build/html/index.html
    

Status for master branch:

[![Build Status](https://travis-ci.org/automl/paramsklearn.svg?branch=master)](https://travis-ci.org/automl/paramsklearn)
[![Code Health](https://landscape.io/github/automl/paramsklearn/master/landscape.png)](https://landscape.io/github/automl/paramsklearn/master)
[![Coverage Status](https://coveralls.io/repos/automl/paramsklearn/badge.svg?branch=master&service=github)](https://coveralls.io/github/automl/paramsklearn?branch=master)

Status for development branch

[![Build Status](https://travis-ci.org/automl/paramsklearn.svg?branch=development)](https://travis-ci.org/automl/paramsklearn)
[![Code Health](https://landscape.io/github/automl/paramsklearn/development/landscape.png)](https://landscape.io/github/automl/paramsklearn/development)
[![Coverage Status](https://coveralls.io/repos/automl/paramsklearn/badge.svg?branch=development&service=github)](https://coveralls.io/github/automl/paramsklearn?branch=development)
