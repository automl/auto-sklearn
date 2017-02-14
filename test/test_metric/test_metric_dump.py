import shutil
import sys
import os
import time
import unittest

from autosklearn.util.backend import create
from autosklearn.metrics import get_metric, get_metric_from_loss
from autosklearn.metrics.factories import KnownMetric, CustomMetric, \
    PackageMetric
from autosklearn.metrics.loss_factory import LossMetricDecorator
from autosklearn.constants import MULTICLASS_CLASSIFICATION


parent_directory = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_directory)


def metric_function(true, prediction):
    return 42.


def loss_function(true, prediction):
    return 42.


class DataManagerStub(object):

    def __init__(self, metric):
        self.info = { 'metric': metric }


class MetricDefinitionsTest(unittest.TestCase):
    _multiprocess_can_split_ = True
    """All tests which are a subclass of this must define their own output
    directory and call self._setUp."""

    def setUp(self):
        self.test_dir = os.path.dirname(__file__)

    def _setUp(self, output):
        if os.path.exists(output):
            for i in range(10):
                try:
                    shutil.rmtree(output)
                    break
                except OSError:
                    time.sleep(1)
        try:
            os.makedirs(output)
        except OSError:
            pass

    def _tearDown(self, output):
        if os.path.exists(output):
            for i in range(10):
                try:
                    shutil.rmtree(output)
                    break
                except OSError:
                    time.sleep(1)

    def test_known_metric(self):
        metric = get_metric('acc', task_type=MULTICLASS_CLASSIFICATION)
        restored_metric = self._dump_and_restore_metric(metric)

        self.assertIsNotNone(restored_metric)
        self.assertIsInstance(restored_metric, KnownMetric)
        self.assertEqual('acc_metric', restored_metric.name)

    def test_package_metric(self):
        package = 'autosklearn.metrics.classification_metrics.acc_metric'
        metric = get_metric(package)
        restored_metric = self._dump_and_restore_metric(metric)

        self.assertIsNotNone(restored_metric)
        self.assertIsInstance(restored_metric, PackageMetric)
        self.assertEqual(package, restored_metric.name)

    def test_custom_metric(self):
        metric = get_metric(metric_function)
        restored_metric = self._dump_and_restore_metric(metric)

        self.assertIsNotNone(restored_metric)
        self.assertIsInstance(restored_metric, CustomMetric)
        self.assertEqual('custom', restored_metric.name)
        self.assertEqual(42, restored_metric.calculate_score(None, None))

    def test_package_loss(self):
        package = 'autosklearn.metrics.classification_metrics.acc_metric'
        metric = get_metric_from_loss(package)
        restored_metric = self._dump_and_restore_metric(metric)

        self.assertIsNotNone(restored_metric)
        self.assertIsInstance(restored_metric, LossMetricDecorator)
        self.assertEqual('reverted_' + package, restored_metric.name)

    def test_custom_loss(self):
        metric = get_metric_from_loss(loss_function)
        restored_metric = self._dump_and_restore_metric(metric)

        self.assertIsNotNone(restored_metric)
        self.assertIsInstance(restored_metric, LossMetricDecorator)
        self.assertEqual('reverted_custom', restored_metric.name)
        self.assertEqual(1- 42, restored_metric.calculate_score(None, None))

    def _dump_and_restore_metric(self, metric):
        output_dir = os.path.join(self.test_dir, '..', '.tmp_metric_definition')
        self._setUp(output_dir)

        backend = create(output_directory=output_dir,
                         temporary_directory=output_dir)
        data_manager = DataManagerStub(metric)
        backend.save_datamanager(data_manager)
        data_manager = backend.load_datamanager()

        self._tearDown(output_dir)

        self.assertIsNotNone(data_manager)

        return data_manager.info['metric']
