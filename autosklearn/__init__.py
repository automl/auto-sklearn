# -*- encoding: utf-8 -*-
import os

import pkg_resources

from autosklearn.estimators import AutoSklearnClassifier
from automl import get_automl_logger
smac = pkg_resources.resource_filename(
    'autosklearn',
    'binaries/smac-v2.08.01-development-1/smac-v2.08.01-development-1/')
runsolver = pkg_resources.resource_filename('autosklearn', 'binaries/')
os.environ['PATH'] = smac + os.pathsep + runsolver + os.pathsep + \
    os.environ['PATH']
