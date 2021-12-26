# NOTE: Only supports linux
#
# A simple makefile to help with small tasks related to development of autosklearn
# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.
# * install-dev - install everything required for dev purposes
# * clean - cleans up any kind of build that happened, doc or dist
# * build - builds the dist
# * publish - publishes a release to pypi
# * test-publish - publishes to testpypi for testing a release
# * doc - Just make docs without examples
# * links - Check link of docs
# * examples - Make docs with examples

.PHONY: help install-dev clean clean-doc clean-build build doc links examples publish

help:
	@echo "Makefile autosklearn"
	@echo "* install-dev      to install dev requirements and init pre-commit"
	@echo "* clean            to clean the dist and doc build files"
	@echo "* build            to build a dist"
	@echo "* doc              to generate and view the html files"
	@echo "* linkcheck        to check the documentation links"
	@echo "* examples         to run and generate the examples"
	@echo "* publish          to help publish the current branch to pypi"

PYTHON ?= python
CYTHON ?= cython
PYTEST ?= python -m pytest
CTAGS ?= ctags
PIP ?= python -m pip
MAKE ?= make

DIR := ${CURDIR}
DIST := ${CURDIR}/dist
DOCDIR := ${DIR}/doc
INDEX_HTML := file://${DOCDIR}/html/build/index.html


install-dev:
	$(PIP) install -e ".[test,examples,docs]"
	pre-commit install

clean-doc:
	$(MAKE) -C ${DOCDIR} clean

clean-build:
	$(PYTHON) setup.py clean
	rm -rf ${DIST}

# Clean up any builds in ./dist as well as doc
clean: clean-doc clean-build

# Build a distribution in ./dist
build:
	$(PYTHON) setup.py bdist

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
publish: clean-build build
	$(PIP) install twine
	$(PYTHON) -m twine upload --repository testpypi ${DIST}/*
	@echo
	@echo "Test with the following line:"
	@echo "pip install --index-url https://test.pypi.org/simple/ auto-sklearn"
	@echo
	@echo "Once you have decided it works, publish to actual pypi with"
	@echo "python -m twine upload dist/*"
