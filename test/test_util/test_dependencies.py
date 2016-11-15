import unittest
import re

from unittest.mock import patch, Mock

import pkg_resources

from autosklearn.util.dependencies import verify_packages, MissingPackageError, \
    IncorrectPackageVersionError


@patch('pkg_resources.get_distribution')
class VerifyPackagesTests(unittest.TestCase):

    def test_existing_package(self, getDistributionMock):
        requirement = 'package'

        verify_packages(requirement)

        getDistributionMock.assert_called_once_with('package')

    def test_missing_package(self, getDistributionMock):
        requirement = 'package'

        getDistributionMock.side_effect = pkg_resources.DistributionNotFound()

        self.assertRaisesRegex(MissingPackageError,
                               "mandatory package 'package' not found", verify_packages, requirement)

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