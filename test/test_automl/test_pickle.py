from __future__ import print_function
from autosklearn.classification import AutoSklearnClassifier
import autosklearn.pipeline.util as putil
import unittest
import six.moves.cPickle as pickle
import os
import sklearn.datasets
import sklearn.metrics
import sklearn.externals.joblib
from base import Base


class PicklingTests(Base, unittest.TestCase):

    def test_can_pickle_classifier(self):
        if self.travis:
            self.skipTest('This test does currently not run on travis-ci. '
                          'Make sure it runs locally on your machine!')

        output = os.path.join(self.test_dir, '..', '.tmp_can_pickle')
        self._setUp(output)

        X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
        automl = AutoSklearnClassifier(time_left_for_this_task=15,
                                       per_run_time_limit=15,
                                       tmp_folder=output,
                                       output_folder=output)
        automl.fit(X_train, Y_train)

        initial_predictions = automl.predict(X_test)
        initial_accuracy = sklearn.metrics.accuracy_score(Y_test, initial_predictions)
        self.assertTrue(initial_accuracy > 0.75)

        # Test pickle
        dump_file = os.path.join(output, 'automl.dump.pkl')

        with open(dump_file, 'wb') as f:
            pickle.dump(automl, f)

        with open(dump_file, 'rb') as f:
            restored_automl = pickle.load(f)

        restored_predictions = restored_automl.predict(X_test)
        restored_accuracy = sklearn.metrics.accuracy_score(Y_test, restored_predictions)
        self.assertTrue(restored_accuracy > 0.75)

        self.assertEqual(initial_accuracy, restored_accuracy)

        # Test joblib
        dump_file = os.path.join(output, 'automl.dump.joblib')

        sklearn.externals.joblib.dump(automl, dump_file)

        restored_automl = sklearn.externals.joblib.load(dump_file)

        restored_predictions = restored_automl.predict(X_test)
        restored_accuracy = sklearn.metrics.accuracy_score(Y_test, restored_predictions)
        self.assertTrue(restored_accuracy > 0.75)

        self.assertEqual(initial_accuracy, restored_accuracy)

