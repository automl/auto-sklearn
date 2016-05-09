# simple makefile to simplify repetitive build env management tasks under posix

PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests
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
    make
    cd ..

test-code: in
	$(NOSETESTS) -s -v tests
test-doc:
	$(NOSETESTS) -s -v doc/*.rst

test-coverage:
	rm -rf coverage .coverage
	$(NOSETESTS) -s -v --with-coverage tests

test: test-code test-sphinxext test-doc
