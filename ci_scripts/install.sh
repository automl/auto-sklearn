if [[ "$DISTRIB" = "conda" ]]; then

    wget $MINICONDA_URL -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    if [[ `which conda` ]]; then echo 'Conda installation successful'; else exit 1; fi
    conda create -n testenv --yes python=$PYTHON_VERSION pip wheel gxx_linux-64 gcc_linux-64 swig
    source activate testenv

else

    sudo apt-get update
    sudo apt install -y swig

fi

