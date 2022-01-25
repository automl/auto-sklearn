:orphan:

.. _installation:

============
Installation
============

System requirements
===================

auto-sklearn has the following system requirements:

* Linux operating system (for example Ubuntu) (`get Linux here <https://www.wikihow.com/Install-Linux>`_)
* Python (>=3.7) (`get Python here <https://www.python.org/downloads/>`_),
* C++ compiler (with C++11 supports) (`get GCC here <https://www.tutorialspoint.com/How-to-Install-Cplusplus-Compiler-on-Linux>`_).

In case you try to install Auto-sklearn on a system where no wheel files for the pyrfr package
are provided (see `here <https://pypi.org/project/pyrfr/#files>`_ for available wheels) you also
need:

* SWIG (`get SWIG here <http://www.swig.org/survey.html>`_).

For an explanation of missing Microsoft Windows and macOS support please
check the Section `Windows/macOS compatibility`_.

Installing auto-sklearn
=======================

You can install *auto-sklearn* with `pip` in the usual manner:

.. code:: bash

    pip3 install auto-sklearn

We recommend installing *auto-sklearn* into a
`virtual environment <https://docs.python-guide.org/dev/virtualenvs/>`_
or an
`Anaconda environment <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

If the ``pip3`` installation command fails, make sure you have the `System requirements`_ installed correctly.

Ubuntu installation
===================

To provide Python 3, a C++11 building environment and the latest SWIG version on Ubuntu,
run:

.. code:: bash

    sudo apt-get install build-essential swig python3-dev


Anaconda installation
=====================

You need to enable conda-forge to install *auto-sklearn* via anaconda. This section explains how to enable conda-forge so
installation can be done with the command `conda install auto-sklearn`. 
Optionally, you can also install *auto-sklearn* with `pip` as detailed in the Section `Installing auto-sklearn`_. 

A common installation problem under recent Linux distribution is the
incompatibility of the compiler version used to compile the Python binary
shipped by AnaConda and the compiler installed by the distribution. This can
be solved by installing the *gcc* compiler shipped with AnaConda (as well as
*swig*):

.. code:: bash

    conda install gxx_linux-64 gcc_linux-64 swig


Conda-forge
~~~~~~~~~~~

Installing `auto-sklearn` from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with:

.. code:: bash

    conda config --add channels conda-forge

    conda config --set channel_priority strict


You must have `conda >=4.9`. To update conda or check your current conda version, please follow the instructions from `the official anaconda documentation <https://docs.anaconda.com/anaconda/install/update-version/>`_ . Once the `conda-forge` channel has been enabled, `auto-sklearn` can be installed with:

.. code:: bash

    conda install auto-sklearn


It is possible to list all of the versions of `auto-sklearn` available on your platform with:

.. code:: bash

    conda search auto-sklearn --channel conda-forge

to read in more details check
`auto sklearn feedstock <https://github.com/conda-forge/auto-sklearn-feedstock>`_.

for more information about Conda forge check
`conda-forge documentations <https://conda-forge.org/docs/>`_.

Source Installation
===================

You can install auto-sklearn directly form source by following the below:

.. code:: bash

    git clone --recurse-submodules git@github.com:automl/auto-sklearn.git
    cd auto-sklearn

    # Install it in editable mode with all optional dependencies
    pip install -e ".[test,doc,examples]"

We use submodules so you will have to make sure the submodule is initialized if you
missed the `--recurse-submodules` option.

.. code:: bash

    git clone git@github.com:automl/auto-sklearn.git
    cd auto-sklearn
    git submodule update --init --recursive

    pip install -e ".[test,doc,examples]"


Windows/macOS compatibility
===========================

Windows
~~~~~~~

*auto-sklearn* relies heavily on the Python module ``resource``. ``resource``
is part of Python's `Unix Specific Services <https://docs.python.org/3/library/unix.html>`_
and not available on a Windows machine. Therefore, it is not possible to run
*auto-sklearn* on a Windows machine.

Possible solutions:

* Windows 10 bash shell (see `431 <https://github.com/automl/auto-sklearn/issues/431>`_ and
  `860 <https://github.com/automl/auto-sklearn/issues/860>`_ for suggestions)
* virtual machine
* docker image


macOS
~~~~~

We currently do not know if *auto-sklearn* works on macOS. There are at least two
issues holding us back from actively supporting macOS:

* The ``resource`` module cannot enforce a memory limit on a Python process
  (see `SMAC3/issues/115 <https://github.com/automl/SMAC3/issues/115>`_).
* Not all dependencies we are using are set up to work on macOS.

In case you're having issues installing the `pyrfr package <https://github.com/automl/random_forest_run>`_, check out
`this installation suggestion on github <https://github.com/automl/auto-sklearn/issues/360#issuecomment-335150470>`_.

Possible other:

* virtual machine
* docker image

Docker Image
============
A Docker image is also provided on dockerhub. To download from dockerhub,
use:

.. code:: bash

    docker pull mfeurer/auto-sklearn:master

You can also verify that the image was downloaded via:

.. code:: bash

    docker images  # Verify that the image was downloaded

This image can be used to start an interactive session as follows:

.. code:: bash

    docker run -it mfeurer/auto-sklearn:master

To start a Jupyter notebook, you could instead run e.g.:

.. code:: bash

    docker run -it -v ${PWD}:/opt/nb -p 8888:8888 mfeurer/auto-sklearn:master /bin/bash -c "mkdir -p /opt/nb && jupyter notebook --notebook-dir=/opt/nb --ip='0.0.0.0' --port=8888 --no-browser --allow-root"

Alternatively, it is possible to use the development version of auto-sklearn by replacing all
occurences of ``master`` by ``development``.
