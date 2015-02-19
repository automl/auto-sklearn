__author__ = 'eggenspk'

import inspect
import os
import pkgutil
import sys

from ..regression_base import ParamSklearnRegressionAlgorithm

regressor_directory = os.path.split(__file__)[0]
_regressors = {}


for module_loader, module_name, ispkg in pkgutil.iter_modules([regressor_directory]):
    full_module_name = "%s.%s" % (__package__, module_name)
    if full_module_name not in sys.modules and not ispkg:
        module = module_loader.find_module(module_name).load_module(full_module_name)

        for member_name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and ParamSklearnRegressionAlgorithm in obj.__bases__:
                # TODO test if the obj implements the interface
                # Keep in mind that this only instantiates the ensemble_wrapper,
                # but not the real target classifier
                classifier = obj
                _regressors[module_name] = classifier
