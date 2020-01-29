:orphan:

.. _installation:

============
Installation
============

System requirements
===================

auto-sklearn has the following system requirements:

* Linux operating system (for example Ubuntu) `(get Linux here) <https://www.wikihow.com/Install-Linux>`_,
* Python (>=3.5) `(get Python here) <https://www.python.org/downloads/>`_.
* C++ compiler (with C++11 supports) `(get GCC here) <https://www.tutorialspoint.com/How-to-Install-Cplusplus-Compiler-on-Linux>`_ and
* SWIG (version 3.0 or later) `(get SWIG here) <http://www.swig.org/survey.html>`_.

For an explanation of missing Microsoft Windows and MAC OSX support please
check the Section `Windows/OSX compatibility`_.

Installing auto-sklearn
=======================

Please install all dependencies manually with:

.. code:: bash

    curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install

Then install *auto-sklearn*:

.. code:: bash

    pip install auto-sklearn

We recommend installing *auto-sklearn* into a
`virtual environment <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_
or an
`Anaconda environment <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

If the ``pip`` installation command fails, make sure you have the `System requirements`_ installed correctly.

Ubuntu installation
===================

To provide a C++11 building environment and the lateste SWIG version on Ubuntu,
run:

.. code:: bash

    sudo apt-get install build-essential swig


Anaconda installation
=====================

Anaconda does not ship *auto-sklearn*, and there are no conda packages for
*auto-sklearn*. Thus, it is easiest to install *auto-sklearn* as detailed in
the Section `Installing auto-sklearn`_.

A common installation problem under recent Linux distribution is the
incompatibility of the compiler version used to compile the Python binary
shipped by AnaConda and the compiler installed by the distribution. This can
be solved by installing the *gcc* compiler shipped with AnaConda (as well as
*swig*):

.. code:: bash

    conda install gxx_linux-64 gcc_linux-64 swig


Windows/OSX compatibility
=========================

Windows
~~~~~~~

*auto-sklearn* relies heavily on the Python module ``resource``. ``resource``
is part of Python's `Unix Specific Services <https://docs.python.org/3/library/unix.html>`_
and not available on a Windows machine. Therefore, it is not possible to run
*auto-sklearn* on a Windows machine.

Possible solutions (not tested):

* Windows 10 bash shell
* virtual machine
* docker image

Mac OSX
~~~~~~~

We currently do not know if *auto-sklearn* works on OSX. There are at least two
issues holding us back from actively supporting OSX:

* The ``resource`` module cannot enforce a memory limit on a Python process
  (see `SMAC3/issues/115 <https://github.com/automl/SMAC3/issues/115>`_).
* OSX machines on `travis-ci <https://travis-ci.org/>`_ take more than 30
  minutes to spawn. This makes it impossible for us to run unit tests for
  *auto-sklearn* and its dependencies `SMAC3 <https://github.com/automl/SMAC3>`_
  and `ConfigSpace <https://github.com/automl/ConfigSpace>`_.

In case you're having issues installing the `pyrfr package <https://github.com/automl/random_forest_run>`_, check out
`this installation suggestion on github <https://github.com/automl/auto-sklearn/issues/360#issuecomment-335150470>`_.

Possible other solutions (not tested):

* virtual machine
* docker image
