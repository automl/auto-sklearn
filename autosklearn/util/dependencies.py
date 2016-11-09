from warnings import warn

import pkg_resources
import re

from distutils.version import LooseVersion


RE_PATTERN = re.compile('^(?P<name>[\w\-]+)((?P<operation>==|>=|>)(?P<version>(\d+\.)?(\d+\.)?(\d+)))?$')


def verify_packages(packages):
    if not packages:
        return
    if isinstance(packages, str):
        packages = packages.splitlines()

    for package in packages:
        if not package:
            continue

        match = RE_PATTERN.match(package)
        if match:
            name = match.group('name')
            operation = match.group('operation')
            version = match.group('version')
            _verify_package(name, operation, version)
        else:
            raise ValueError('Unable to read requirement: %s' % package)


def _verify_package(name, operation, version):
    try:
        module = pkg_resources.get_distribution(name)
    except pkg_resources.DistributionNotFound:
        warn('mandatory package \'%s\' not found' % name)
        return

    if not operation:
        return

    required_version = LooseVersion(version)
    installed_version = LooseVersion(module.version)

    if operation == '==':
        check = required_version == installed_version
    elif operation == '>':
        check = installed_version > required_version
    elif operation == '>=':
        check = installed_version > required_version or \
                installed_version == required_version
    else:
        raise NotImplementedError('operation %s is not supported' % operation)
    if not check:
        warn('\'%s\' version mismatch (%s%s)' % (name, operation, required_version))
