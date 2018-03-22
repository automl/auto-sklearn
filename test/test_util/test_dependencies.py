import unittest
import pkg_resources
import re

from unittest.mock import patch, Mock

import numpy as np

from autosklearn.util.dependencies import verify_packages, MissingPackageError, \
    IncorrectPackageVersionError


@patch('pkg_resources.get_distribution')
class VerifyPackagesTests(unittest.TestCase):

    def test_existing_package(self, getDistributionMock):
        requirement = 'package'
        distribution_mock = unittest.mock.Mock()
        getDistributionMock.return_value = distribution_mock
        distribution_mock.version = '1.0.0'

        verify_packages(requirement)

        getDistributionMock.assert_called_once_with('package')

    def test_missing_package(self, getDistributionMock):
        requirement = 'package'

        getDistributionMock.side_effect = pkg_resources.DistributionNotFound()

        self.assertRaisesRegex(
            MissingPackageError,
            "Mandatory package 'package' not found",
            verify_packages,
            requirement,
        )

    @patch('importlib.import_module')
    def test_package_can_only_be_imported(self, import_mock, getDistributionMock):

        getDistributionMock.side_effect = pkg_resources.DistributionNotFound()
        package = unittest.mock.Mock()
        package.__version__ = np.__version__
        import_mock.return_value = package

        verify_packages('numpy')

    def test_correct_package_versions(self, getDistributionMock):
        requirement = 'package==0.1.2\n' \
                      'package>0.1\n' \
                      'package>=0.1'

        moduleMock = Mock()
        moduleMock.version = '0.1.2'
        getDistributionMock.return_value = moduleMock

        verify_packages(requirement)

        getDistributionMock.assert_called_with('package')
        self.assertEqual(3, len(getDistributionMock.call_args_list))

    def test_wrong_package_version(self, getDistributionMock):
        requirement = 'package>0.1.2'

        moduleMock = Mock()
        moduleMock.version = '0.1.2'
        getDistributionMock.return_value = moduleMock

        self.assertRaisesRegex(IncorrectPackageVersionError,
                               re.escape("'package 0.1.2' version mismatch (>0.1.2)"), verify_packages, requirement)

    def test_outdated_requirement(self, getDistributionMock):
        requirement = 'package>=0.1'

        moduleMock = Mock()
        moduleMock.version = '0.0.9'
        getDistributionMock.return_value = moduleMock

        self.assertRaisesRegex(IncorrectPackageVersionError,
                               re.escape("'package 0.0.9' version mismatch (>=0.1)"), verify_packages, requirement)

    def test_too_fresh_requirement(self, getDistributionMock):
        requirement = 'package==0.1.2'

        moduleMock = Mock()
        moduleMock.version = '0.1.3'
        getDistributionMock.return_value = moduleMock

        self.assertRaisesRegex(IncorrectPackageVersionError,
                               re.escape("'package 0.1.3' version mismatch (==0.1.2)"), verify_packages, requirement)
