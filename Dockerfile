FROM ubuntu:20.04

WORKDIR /auto-sklearn

# install linux packages
RUN apt-get update

# Set the locale
# workaround for https://github.com/automl/auto-sklearn/issues/867
RUN apt-get -y install locales
RUN touch /usr/share/locale/locale.alias
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# set environment variables to only use one core
RUN export OPENBLAS_NUM_THREADS=1
RUN export MKL_NUM_THREADS=1
RUN export BLAS_NUM_THREADS=1
RUN export OMP_NUM_THREADS=1

# install build requirements
RUN apt install -y python3-dev python3-pip
RUN pip3 install --upgrade setuptools
RUN apt install -y build-essential

RUN apt install -y swig

# Copy the checkout autosklearn version for installation
ADD . /auto-sklearn/

# Upgrade pip then install dependencies
RUN pip3 install --upgrade pip
RUN pip3 install pytest==4.6.* pep8 codecov pytest-cov flake8 flaky openml
RUN cat /auto-sklearn/requirements.txt | xargs -n 1 -L 1 pip3 install
RUN pip3 install jupyter

# Install
RUN pip3 install /auto-sklearn/
