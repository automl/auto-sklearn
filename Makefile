# NOTE: Used on linux, limited support outside of Linux
#
# A modern makefile for auto-sklearn development using Poetry
# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

.PHONY: help install install-dev check format pre-commit clean clean-doc clean-build build doc links examples publish test test-cov benchmark

help:
	@echo "Makefile auto-sklearn"
	@echo "* install          to install the package"
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
	@echo "* test-cov         to run the tests with coverage"
	@echo "* benchmark        to run performance benchmarks"

PYTHON ?= python
CYTHON ?= cython
PYTEST ?= poetry run pytest
CTAGS ?= ctags
POETRY ?= poetry
MAKE ?= make
BLACK ?= poetry run black
ISORT ?= poetry run isort
PYDOCSTYLE ?= poetry run pydocstyle
MYPY ?= poetry run mypy
PRECOMMIT ?= poetry run pre-commit
FLAKE8 ?= poetry run flake8

DIR := "${CURDIR}"
DIST := "${CURDIR}/dist"
DOCDIR := "${DIR}/doc"
INDEX_HTML := "file://${DOCDIR}/build/html/index.html"

install:
	$(POETRY) install --no-dev

install-dev:
	$(POETRY) install
	$(POETRY) run pre-commit install

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
	$(POETRY) build

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

# Publish to PyPI
publish: clean build
	$(POETRY) publish
	@echo
	@echo "Package published to PyPI!"

test:
	$(PYTEST) test

test-cov:
	$(PYTEST) test --cov=autosklearn --cov-report=html --cov-report=term

benchmark:
	$(PYTEST) test --benchmark-autosave --benchmark-save=current
