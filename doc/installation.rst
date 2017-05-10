:orphan:

.. _installation:

============
Installation
============

**Prerequisities**: *auto-sklearn* is written in python and was developed
with Ubuntu. It should run on other Linux distributions, but won't work on a MAC
or on a windows PC. We aim to always support the two latests python versions,
which are 3.4 and 3.5 at the moment. It is built around scikit-learn 0.17.1 and
needs a compiler for C++ 11.

Please install all dependencies manually with:

.. code:: bash

    curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install

Then install *auto-sklearn*

.. code:: bash

    pip install auto-sklearn

We recommend installing *auto-sklearn* into a `virtual environment
<http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.