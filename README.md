### Download and build the documentation ###

    git clone https://bitbucket.org/mfeurer/autosklearn.git
    cd autosklearn
    python setup.py
    
To build the documentation you also need the packages `sphinx` and `numpydoc`.

    pip install sphinx
    pip install numpydoc
    make html
    firefox `pwd`/build/html/index.html
