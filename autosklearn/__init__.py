# -*- encoding: utf-8 -*-
from autosklearn.util import dependencies

__version__ = '0.1.1'

__MANDATORY_PACKAGES__ = '''
scikit-learn==0.17.1
smac==0.2.1
lockfile>=0.10
ConfigSpace>=0.2.1
pyrfr==0.2.0
'''

dependencies.verify_packages(__MANDATORY_PACKAGES__)
