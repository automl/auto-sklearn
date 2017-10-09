# -*- encoding: utf-8 -*-
from autosklearn.util import dependencies
from autosklearn.__version__ import __version__


__MANDATORY_PACKAGES__ = '''
numpy>=1.9
scikit-learn>=0.18.1,<0.19
lockfile>=0.10
smac>=0.6.0,<0.7
pyrfr>=0.6.1,<0.7
ConfigSpace>=0.3.3,<0.4
pyrfr>=0.6.0,<0.7
'''

dependencies.verify_packages(__MANDATORY_PACKAGES__)
