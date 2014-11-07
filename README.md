### Download and build the documentation ###

    git clone https://bitbucket.org/mfeurer/autosklearn.git
    cd autosklearn
    python setup.py install

Installation with `pip`

    pip install numpy scipy scikit-learn==0.15.1 numpydoc sphinx
    pip install --editable git+https://bitbucket.org/mfeurer/autosklearn#egg=AutoSklearn

To build the documentation you also need the packages `sphinx` and `numpydoc`.

    pip install sphinx
    pip install numpydoc
    make html
    firefox `pwd`/build/html/index.html
