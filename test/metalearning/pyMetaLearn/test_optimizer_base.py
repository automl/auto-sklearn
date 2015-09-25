from collections import OrderedDict
import unittest

from autosklearn.metalearning.optimizers import optimizer_base


class OptimizerBaseTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        self.hyperparameters = OrderedDict()
        self.hyperparameters["x"] = [-5, 0, 5, 10]
        self.hyperparameters["y"] = [0, 5, 10, 15]

    def test_parse_hyperopt_string(self):
        hyperparameter_string = "x {-5, 0, 5, 10}\ny {0, 5, 10, 15}"
        expected = OrderedDict([["x", ["-5", "0", "5", "10"]],
                                ["y", ["0", "5", "10", "15"]]])
        ret = optimizer_base.parse_hyperparameter_string(hyperparameter_string)
        self.assertEqual(ret, expected)

        hyperparameter_string = "x {-5, 0, 5, 10} [5]\ny {0, 5, 10, 15}"
        ret = optimizer_base.parse_hyperparameter_string(hyperparameter_string)
        self.assertEqual(ret, expected)

        hyperparameter_string = "x {-5, 0, 5, 10}\ny {0, 5, 10, 15} [5]"
        ret = optimizer_base.parse_hyperparameter_string(hyperparameter_string)
        self.assertEqual(ret, expected)

        hyperparameter_string = "x {-5, 0, 5, 10}\ny 0, 5, 10, 15} [5]"
        self.assertRaises(ValueError, optimizer_base.parse_hyperparameter_string,
                          hyperparameter_string)

    def test_construct_cli_call(self):
        cli_call = optimizer_base.construct_cli_call("cv.py", {"x": -5, "y": 0})
        self.assertEqual(cli_call, "cv.py -x \"'-5'\" -y \"'0'\"")