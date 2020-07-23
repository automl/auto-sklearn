FROM ubuntu:18.04

WORKDIR /auto-sklearn

# Copy the checkout autosklearn version for installation
ADD . /auto-sklearn/

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

RUN apt install -y python3-dev python3-pip
RUN pip3 install --upgrade setuptools
RUN apt-get install -y build-essential curl 

# https://github.com/automl/auto-sklearn/issues/314
RUN apt-get install -y swig3.0
RUN ln -s /usr/bin/swig3.0 /usr/bin/swig

# Upgrade pip then install dependencies
RUN pip3 install --upgrade pip
RUN pip3 install pytest==4.6.* pep8 codecov pytest-cov flake8 flaky openml
RUN curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip3 install
RUN pip3 install jupyter

# Install
RUN pip3 install -e /auto-sklearn/
