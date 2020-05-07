# simple makefile to simplify repetitive build env management tasks under posix

PYTHON ?= python
CYTHON ?= cython
PYTEST ?= python -m pytest
CTAGS ?= ctags

all: clean inplace test

clean:
	$(PYTHON) setup.py clean
	rm -rf dist

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

doc:
	cd ./doc
	make html
	cd ..

test-code: in
	$(PYTEST) -s -v test
test-doc:
	$(PYTEST) -s -v doc/*.rst

test-coverage:
	rm -rf coverage .coverage
	$(PYTEST) -s -v --with-coverage test

test: test-code test-sphinxext test-doc
