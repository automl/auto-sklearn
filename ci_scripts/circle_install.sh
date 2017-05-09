#!bin/bash

# on circle ci, each command run with it's own execution context so we have to
# activate the conda testenv on a per command basis. That's why we put calls to
# python (conda) in a dedicated bash script and we activate the conda testenv
# here.
source activate testenv

# install documentation building dependencies
pip install --upgrade numpy
pip install --upgrade matplotlib setuptools nose coverage sphinx pillow sphinx-gallery sphinx_bootstrap_theme cython numpydoc
# And finally, all other dependencies
cat requirements.txt | xargs -n 1 -L 1 pip install

python setup.py clean
python setup.py develop

# pipefail is necessary to propagate exit codes
set -o pipefail && cd doc && make html 2>&1 | tee ~/log.txt