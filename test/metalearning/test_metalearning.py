import os
import unittest
from AutoSklearn.util import get_dataset
from HPOlibConfigSpace.converters import pcs_parser

from AutoML2015.metalearning import metalearning


class Test(unittest.TestCase):
    def test_metalearning(self):
        X_train, Y_train, X_test, Y_test = get_dataset('iris')

        eliminate_class_two = Y_train != 2
        X_train = X_train[eliminate_class_two]
        Y_train = Y_train[eliminate_class_two]

        with open(os.path.join(os.path.dirname(metalearning.__file__),
                               "files", "params.pcs")) as fh:
            configuration_space = pcs_parser.read(fh)

        initial_configuration_strings_for_smac = \
            metalearning.create_metalearning_string_for_smac_call(
            X_train, Y_train, configuration_space, None, "iris", "bac_metric")

        print initial_configuration_strings_for_smac