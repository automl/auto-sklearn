# -*- encoding: utf-8 -*-
from autosklearn.util import dependencies
from autosklearn.__version__ import __version__


__MANDATORY_PACKAGES__ = '''
numpy>=1.9,<1.12
scikit-learn==0.17.1
smac==0.3.0
lockfile>=0.10
ConfigSpace>=0.3.1,<0.4
pyrfr==0.2.0
xgboost==0.4a30
'''

dependencies.verify_packages(__MANDATORY_PACKAGES__)
