:orphan:

.. _installation:

============
Installation
============

System requirements
===================

auto-sklearn has the following system requirements:

* Linux operating system (for example Ubuntu),
* Python (>=3.4).
* C++ compiler (which supports C++11) and SWIG

For an explanation of missing Microsoft Windows and MAC OSX support please
check the Section `Windows/OSX compabilities`_.

Python requirements
===================

Please install all dependencies manually with:

.. code:: bash

    curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install

Then install *auto-sklearn*

.. code:: bash

    pip install auto-sklearn

We recommend installing *auto-sklearn* into a `virtual environment
<http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_ or an `Anaconda
environment <https://conda.io/docs/using/envs.html>`_.

Anaconda installation
=====================

Anaconda does not ship *auto-sklearn*, and there are no conda packages for
*auto-sklearn*. Thus, it is easiest to install *auto-sklearn* as detailed in
the Section `Python requirements`_.

A common installation problem under recent Linux distribution is the
incompability of the compiler version used to compile the Python binary
shipped by AnaConda and the compiler installed by the distribution. This can
be solved by istalling the *gcc* compiler shipped with AnaConda (as well as
*swig*):

.. code:: bash

    conda install gcc swig


Windows/OSX compabilities
=========================

