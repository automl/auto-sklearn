### Download and build the documentation ###

    pip install git+https://github.com/mfeurer/HPOlibConfigSpace#egg=HPOlibConfigSpace0.1dev
    git clone https://bitbucket.org/mfeurer/autosklearn.git
    cd autosklearn
    python setup.py install
    python setup.py test

Installation with `pip`

    pip install numpy scipy scikit-learn==0.15.2 numpydoc sphinx
    pip install git+https://github.com/mfeurer/HPOlibConfigSpace#egg=HPOlibConfigSpace0.1dev
    pip install --editable git+https://bitbucket.org/mfeurer/autosklearn#egg=AutoSklearn

To build the documentation you also need the packages `sphinx` and `numpydoc`.

    pip install sphinx
    pip install numpydoc
    make html
    firefox `pwd`/build/html/index.html