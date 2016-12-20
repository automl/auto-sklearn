# -*- encoding: utf-8 -*-
from autosklearn.util import dependencies
from autosklearn.__version__ import __version__


__MANDATORY_PACKAGES__ = '''
scikit-learn==0.17.1
smac==0.2.2
lockfile>=0.10
ConfigSpace>=0.2.1
pyrfr==0.2.0
xgboost==0.4a30
'''

dependencies.verify_packages(__MANDATORY_PACKAGES__)
