import unittest
from pyMetaLearn.metalearning.test_meta_base import MetaBaseTest
from pyMetaLearn.metalearning.kNearestDatasets.test_kND import kNDTest


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(unittest.makeSuite(MetaBaseTest))
    _suite.addTest(unittest.makeSuite(kNDTest))
    return _suite


if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite())