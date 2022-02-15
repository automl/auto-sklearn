# NOTE: Used on linux, limited support outside of Linux
#
# A simple makefile to help with small tasks related to development of autosklearn
# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

.PHONY: help install-dev check format pre-commit clean clean-doc clean-build build doc links examples publish test

help:
	@echo "Makefile autosklearn"
	@echo "* install-dev      to install all dev requirements and install pre-commit"
	@echo "* check            to check the source code for issues"
	@echo "* format           to format the code with black and isort"
	@echo "* pre-commit       to run the pre-commit check"
	@echo "* clean            to clean the dist and doc build files"
	@echo "* build            to build a dist"
	@echo "* doc              to generate and view the html files"
	@echo "* linkcheck        to check the documentation links"
	@echo "* examples         to run and generate the examples"
	@echo "* publish          to help publish the current branch to pypi"
	@echo "* test             to run the tests"

PYTHON ?= python
CYTHON ?= cython
PYTEST ?= python -m pytest
CTAGS ?= ctags
PIP ?= python -m pip
MAKE ?= make
BLACK ?= black
ISORT ?= isort
PYDOCSTYLE ?= pydocstyle
MYPY ?= mypy
PRECOMMIT ?= pre-commit
FLAKE8 ?= flake8

DIR := "${CURDIR}"
DIST := "${CURDIR}/dist""
DOCDIR := "${DIR}/doc"
INDEX_HTML := "file://${DOCDIR}/build/html/index.html"

install-dev:
	$(PIP) install -e ".[test,examples,docs]"
	pre-commit install

check-black:
	$(BLACK) autosklearn examples test --check || :

check-isort:
	$(ISORT) autosklearn test --check || :

check-pydocstyle:
	$(PYDOCSTYLE) autosklearn || :

check-mypy:
	$(MYPY) autosklearn || :

check-flake8:
	$(FLAKE8) autosklearn || :
	$(FLAKE8) test || :

# pydocstyle does not have easy ignore rules, instead, we include as they are covered
check: check-black check-isort check-mypy check-flake8 # check-pydocstyle

pre-commit:
	$(PRECOMMIT) run --all-files

format-black:
	$(BLACK) autosklearn/.*
	$(BLACK) test/.*
	$(BLACK) examples/.*

format-isort:
	$(ISORT) autosklearn
	$(ISORT) test


format: format-black format-isort

clean-doc:
	$(MAKE) -C ${DOCDIR} clean

clean-build:
	$(PYTHON) setup.py clean
	rm -rf ${DIST}

# Clean up any builds in ./dist as well as doc
clean: clean-doc clean-build

# Build a distribution in ./dist
build:
	$(PYTHON) setup.py sdist

doc:
	$(MAKE) -C ${DOCDIR} html-noexamples
	@echo
	@echo "View docs at:"
	@echo ${INDEX_HTML}

links:
	$(MAKE) -C ${DOCDIR} linkcheck

examples:
	$(MAKE) -C ${DOCDIR} html
	@echo
	@echo "View docs at:"
	@echo ${INDEX_HTML}

# Publish to testpypi
# Will echo the commands to actually publish to be run to publish to actual PyPi
# This is done to prevent accidental publishing but provide the same conveniences
publish: clean build
	$(PIP) install twine
	$(PYTHON) -m twine upload --repository testpypi ${DIST}/*
	@echo
	@echo "Test with the following:"
	@echo "* Create a new virtual environment to install the uplaoded distribution into"
	@echo "* Run the following:"
	@echo
	@echo "        pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ autosklearn"
	@echo
	@echo "* Run this to make sure it can import correctly, plus whatever else you'd like to test:"
	@echo
	@echo "        python -c 'import autosklearn'"
	@echo
	@echo "Once you have decided it works, publish to actual pypi with"
	@echo
	@echo "    python -m twine upload dist/*"

test:
	$(PYTEST) test
