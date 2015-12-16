import collections
import importlib
import inspect
import os
import pkgutil
import sys

from ..base import AutoSklearnPreprocessingAlgorithm
from .rescaling import RescalingChoice


preprocessors_directory = os.path.split(__file__)[0]
_preprocessors = collections.OrderedDict()

for module_loader, module_name, ispkg in pkgutil.iter_modules(
        [preprocessors_directory]):
    full_module_name = "%s.%s" % (__package__, module_name)
    if full_module_name not in sys.modules and not ispkg:
        module = importlib.import_module(full_module_name)

        for member_name, obj in inspect.getmembers(module):
            if inspect.isclass(
                    obj) and AutoSklearnPreprocessingAlgorithm in obj.__bases__:
                # TODO test if the obj implements the interface
                # Keep in mind that this only instantiates the ensemble_wrapper,
                # but not the real target classifier
                preprocessor = obj
                _preprocessors[module_name] = preprocessor

_preprocessors['rescaling'] = RescalingChoice