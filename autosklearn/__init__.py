# -*- encoding: utf-8 -*-
import os

from autosklearn.estimators import AutoSklearnClassifier
from automl import get_automl_logger
from autosklearn.util.submit_process import get_prog_path

runsolver = os.path.dirname(get_prog_path('runsolver'))
smac = os.path.dirname(get_prog_path('smac'))


os.environ['PATH'] = smac + os.pathsep + runsolver + os.pathsep + \
    os.environ['PATH']
