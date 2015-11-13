# -*- encoding: utf-8 -*-
import os
import shutil
import subprocess
import sys
import tarfile
import urllib

import setuptools
from setuptools.extension import Extension
from setuptools.command.install import install

SMAC_DOWNLOAD_LOCATION = 'http://aad.informatik.uni-freiburg.de/~feurerm/'
SMAC_TAR_NAME = 'smac-v2.08.01-development-1.tar.gz'
# METADATA_LOCATION = "http://aad.informatik.uni-freiburg.de/~feurerm/"
# METADATA_TAR_NAME = "metadata_automl1_000.tar.gz"
RUNSOLVER_LOCATION = 'http://www.cril.univ-artois.fr/~roussel/runsolver/'
RUNSOLVER_TAR_NAME = 'runsolver-3.3.4.tar.bz2'
DOWNLOAD_DIRECTORY = os.path.join(os.path.dirname(__file__), '.downloads')
BINARIES_DIRECTORY = 'autosklearn/binaries'
METADATA_DIRECTORY = 'autosklearn/metalearning/files'


extensions = [Extension('autosklearn.data.competition_c_functions',
                        sources=[
                            'autosklearn/data/competition_c_functions.pyx'
                        ])
             ]


class Download(install):

    def run(self):
        try:
            shutil.rmtree(DOWNLOAD_DIRECTORY)
        except Exception:
            pass

        try:
            os.makedirs(DOWNLOAD_DIRECTORY)
        except Exception:
            pass

        for download_url, filename in [
            (SMAC_DOWNLOAD_LOCATION, SMAC_TAR_NAME),
            # (METADATA_LOCATION, METADATA_TAR_NAME),
            (RUNSOLVER_LOCATION, RUNSOLVER_TAR_NAME)
        ]:
            # This can fail ungracefully, because having these files is
            # crucial to AutoSklearn!
            urllib.urlretrieve(
                os.path.join(download_url, filename),
                filename=os.path.join(DOWNLOAD_DIRECTORY, filename))

            tfile = tarfile.open(os.path.join(DOWNLOAD_DIRECTORY, filename))
            tfile.extractall(os.path.join(
                DOWNLOAD_DIRECTORY,
                filename.replace('.tar.gz', '').replace('.tar.bz2', '')))

        # Build the runsolver
        sys.stdout.write('Building runsolver\n')
        cur_pwd = os.getcwd()
        runsolver_source_path = os.path.join(DOWNLOAD_DIRECTORY,
                                             'runsolver-3.3.4', 'runsolver',
                                             'src')
        os.chdir(runsolver_source_path)
        subprocess.check_call('make')
        os.chdir(cur_pwd)

        # Create a fresh binaries directory
        try:
            shutil.rmtree(BINARIES_DIRECTORY)
        except Exception:
            pass

        try:
            os.makedirs(BINARIES_DIRECTORY)
            with open(os.path.join(BINARIES_DIRECTORY, '__init__.py')):
                pass
        except Exception:
            pass

        # Copy the runsolver into the sources so it gets copied
        shutil.move(os.path.join(runsolver_source_path, 'runsolver'),
                    os.path.join(BINARIES_DIRECTORY, 'runsolver'))

        # Copy SMAC
        shutil.move(os.path.join(DOWNLOAD_DIRECTORY,
                                 SMAC_TAR_NAME.replace('.tar.gz', '')),
                    BINARIES_DIRECTORY)

        # try:
        #    shutil.rmtree(METADATA_DIRECTORY)
        # except Exception:
        #    pass

        # Copy the metadata
        # shutil.move(os.path.join(DOWNLOAD_DIRECTORY,
        #                         METADATA_TAR_NAME.replace(".tar.gz", ""),
        #                         "files"),
        #            METADATA_DIRECTORY)

        # TODO: Normally one wants to call run(self), but this runs distutils and ignores install_requirements for unknown reasons
        # if anyone knows a better way, feel free to change
        install.do_egg_install(self)

        # shutil.rmtree(os.path.join(METADATA_DIRECTORY))
        shutil.rmtree(BINARIES_DIRECTORY)
        shutil.rmtree(DOWNLOAD_DIRECTORY)


setuptools.setup(
    name='AutoSklearn',
    description='Code to participate in the AutoML 2015 challenge.',
    version='0.0.1dev',
    ext_modules=extensions,
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=['setuptools',
                      'numpy>=0.16.0',
                      'psutil',
                      'pyyaml',
                      'scipy>=0.14.0',
                      'scikit-learn==0.16.1',
                      'nose',
                      'lockfile',
                      'HPOlibConfigSpace',
                      'ParamSklearn',
                      'six',
                      'ConfigArgParse',
                      'liac-arff',
                      'pandas'],
    test_suite='nose.collector',
    cmdclass={'install': Download},
    scripts=['scripts/autosklearn'],
    include_package_data=True,
    author='Matthias Feurer',
    author_email='feurerm@informatik.uni-freiburg.de',
    license='BSD',
    platforms=['Linux'],
    classifiers=[],
    url='www.automl.org')
