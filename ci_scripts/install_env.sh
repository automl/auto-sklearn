python --version

if [[ "$DISTRIB" == "conda" ]]; then

    wget $MINICONDA_URL -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    # check if Conda was installed
    if [[ `which conda` ]]; then echo 'Conda installation successful'; else exit 1; fi
    conda create -n testenv --yes pip wheel gxx_linux-64 gcc_linux-64 swig python="$PYTHON"
    source activate testenv

else

    sudo apt install -y python-dev python-pip
    pip install --upgrade setuptools

    # install linux packages
    sudo apt-get update
    # https://github.com/automl/auto-sklearn/issues/314
    sudo apt-get remove swig
    sudo apt-get install swig3.0
    sudo ln -s /usr/bin/swig3.0 /usr/bin/swig

fi
