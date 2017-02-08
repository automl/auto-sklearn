import copy
import queue
import multiprocessing
import os
import sys
import unittest
import unittest.mock

from ConfigSpace import Configuration
import numpy as np
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit
from smac.tae.execute_ta_run import StatusType

from autosklearn.evaluation import get_last_result, TrainEvaluator, eval_holdout, \
    eval_iterative_holdout, eval_cv, eval_partial_cv
from autosklearn.util import backend
from autosklearn.util.pipeline import get_configuration_space
from autosklearn.constants import *

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import get_regression_datamanager, BaseEvaluatorTest, \
    get_binary_classification_datamanager, get_dataset_getters, \
    get_multiclass_classification_datamanager


class Dummy(object):
    def __init__(self):
        self.name = 'dummy'


class TestTrainEvaluator(BaseEvaluatorTest, unittest.TestCase):
    _multiprocess_can_split_ = True

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_holdout(self, pipeline_mock):
        D = get_binary_classification_datamanager()
        D.name = 'test'
        kfold = ShuffleSplit(n=len(D.data['Y_train']), random_state=1, n_iter=1)

        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        output_dir = os.path.join(os.getcwd(), '.test_holdout')

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(D, backend_api, queue_,
                                   configuration=configuration,
                                   cv=kfold,
                                   with_predictions=True,
                                   all_scoring_functions=False,
                                   output_y_test=True)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, None)

        evaluator.fit_predict_and_loss()

        duration, result, seed, run_info, status = evaluator.queue.get(timeout=1)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(result, 1.0)
        self.assertEqual(pipeline_mock.fit.call_count, 1)
        # three calls because of the holdout, the validation and the test set
        self.assertEqual(pipeline_mock.predict_proba.call_count, 3)
        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], 7)
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0], D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0], D.data['Y_test'].shape[0])
        self.assertEqual(evaluator.model.fit.call_count, 1)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_iterative_holdout(self, pipeline_mock):
        # Regular fitting
        D = get_binary_classification_datamanager()
        D.name = 'test'
        kfold = ShuffleSplit(n=len(D.data['Y_train']), random_state=1, n_iter=1)

        class SideEffect(object):
            def __init__(self):
                self.fully_fitted_call_count = 0

            def configuration_fully_fitted(self):
                self.fully_fitted_call_count += 1
                return self.fully_fitted_call_count > 5

        Xt_fixture = 'Xt_fixture'
        pipeline_mock.estimator_supports_iterative_fit.return_value = True
        pipeline_mock.configuration_fully_fitted.side_effect = SideEffect().configuration_fully_fitted
        pipeline_mock.pre_transform.return_value = Xt_fixture, {}
        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        output_dir = os.path.join(os.getcwd(), '.test_iterative_holdout')

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(D, backend_api, queue_,
                                   configuration=configuration,
                                   cv=kfold,
                                   with_predictions=True,
                                   all_scoring_functions=False,
                                   output_y_test=True)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, None)

        class LossSideEffect(object):
            def __init__(self):
                self.losses = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
                self.iteration = 0

            def side_effect(self, *args):
                self.iteration += 1
                return self.losses[self.iteration]
        evaluator._loss = unittest.mock.Mock()
        evaluator._loss.side_effect = LossSideEffect().side_effect

        evaluator.fit_predict_and_loss(iterative=True)
        self.assertEqual(evaluator.file_output.call_count, 5)

        for i in range(1, 6):
            duration, result, seed, run_info, status = evaluator.queue.get(timeout=1)
            self.assertAlmostEqual(result, 1.0 - (0.2 * i))
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(pipeline_mock.iterative_fit.call_count, 5)
        self.assertEqual([cal[1]['n_iter'] for cal in pipeline_mock.iterative_fit.call_args_list], [2, 4, 8, 16, 32])
        # fifteen calls because of the holdout, the validation and the test set
        # and a total of five calls because of five iterations of fitting
        self.assertEqual(evaluator.model.predict_proba.call_count, 15)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], 7)
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0], D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0], D.data['Y_test'].shape[0])
        self.assertEqual(evaluator.file_output.call_count, 5)
        self.assertEqual(evaluator.model.fit.call_count, 0)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_iterative_holdout_interuption(self, pipeline_mock):
        # Regular fitting
        D = get_binary_classification_datamanager()
        D.name = 'test'
        kfold = ShuffleSplit(n=len(D.data['Y_train']), random_state=1, n_iter=1)

        class SideEffect(object):
            def __init__(self):
                self.fully_fitted_call_count = 0

            def configuration_fully_fitted(self):
                self.fully_fitted_call_count += 1
                if self.fully_fitted_call_count == 3:
                    raise ValueError()
                return self.fully_fitted_call_count > 5

        Xt_fixture = 'Xt_fixture'
        pipeline_mock.estimator_supports_iterative_fit.return_value = True
        pipeline_mock.configuration_fully_fitted.side_effect = SideEffect().configuration_fully_fitted
        pipeline_mock.pre_transform.return_value = Xt_fixture, {}
        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        output_dir = os.path.join(os.getcwd(), '.test_iterative_holdout_interuption')

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(D, backend_api, queue_,
                                   configuration=configuration,
                                   cv=kfold,
                                   with_predictions=True,
                                   all_scoring_functions=False,
                                   output_y_test=True)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, None)

        class LossSideEffect(object):
            def __init__(self):
                self.losses = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
                self.iteration = 0

            def side_effect(self, *args):
                self.iteration += 1
                return self.losses[self.iteration]
        evaluator._loss = unittest.mock.Mock()
        evaluator._loss.side_effect = LossSideEffect().side_effect

        self.assertRaises(ValueError, evaluator.fit_predict_and_loss, iterative=True)
        self.assertEqual(evaluator.file_output.call_count, 2)

        for i in range(1, 3):
            duration, result, seed, run_info, status = evaluator.queue.get(timeout=1)
            self.assertAlmostEqual(result, 1.0 - (0.2 * i))
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(pipeline_mock.iterative_fit.call_count, 2)
        # fifteen calls because of the holdout, the validation and the test set
        # and a total of five calls because of five iterations of fitting
        self.assertEqual(evaluator.model.predict_proba.call_count, 6)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], 7)
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0], D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0], D.data['Y_test'].shape[0])
        self.assertEqual(evaluator.file_output.call_count, 2)
        self.assertEqual(evaluator.model.fit.call_count, 0)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_iterative_holdout_not_iterative(self, pipeline_mock):
        # Regular fitting
        D = get_binary_classification_datamanager()
        D.name = 'test'
        kfold = ShuffleSplit(n=len(D.data['Y_train']), random_state=1, n_iter=1)

        Xt_fixture = 'Xt_fixture'
        pipeline_mock.estimator_supports_iterative_fit.return_value = False
        pipeline_mock.pre_transform.return_value = Xt_fixture, {}
        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        output_dir = os.path.join(os.getcwd(), '.test_iterative_holdout_not_iterative')

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(D, backend_api, queue_,
                                   configuration=configuration,
                                   cv=kfold,
                                   with_predictions=True,
                                   all_scoring_functions=False,
                                   output_y_test=True)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, None)

        evaluator.fit_predict_and_loss(iterative=True)
        self.assertEqual(evaluator.file_output.call_count, 1)

        duration, result, seed, run_info, status = evaluator.queue.get(timeout=1)
        self.assertAlmostEqual(result, 1.0)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(pipeline_mock.iterative_fit.call_count, 0)
        # fifteen calls because of the holdout, the validation and the test set
        # and a total of five calls because of five iterations of fitting
        self.assertEqual(evaluator.model.predict_proba.call_count, 3)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], 7)
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0], D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0], D.data['Y_test'].shape[0])
        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(evaluator.model.fit.call_count, 1)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_cv(self, pipeline_mock):
        D = get_binary_classification_datamanager()
        kfold = StratifiedKFold(y=D.data['Y_train'].flatten(), random_state=1,
                                n_folds=5, shuffle=True)

        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        output_dir = os.path.join(os.getcwd(), '.test_cv')

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(D, backend_api, queue_,
                                   configuration=configuration,
                                   cv=kfold,
                                   with_predictions=True,
                                   all_scoring_functions=False,
                                   output_y_test=True)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, None)

        evaluator.fit_predict_and_loss()

        duration, result, seed, run_info, status = evaluator.queue.get(timeout=1)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(evaluator.file_output.call_count, 1)
        self.assertEqual(result, 1.0)
        self.assertEqual(pipeline_mock.fit.call_count, 5)
        # Fifteen calls because of the holdout, the validation and the test set
        self.assertEqual(pipeline_mock.predict_proba.call_count, 15)
        self.assertEqual(evaluator.file_output.call_args[0][0].shape[0], D.data['Y_train'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][1].shape[0], D.data['Y_valid'].shape[0])
        self.assertEqual(evaluator.file_output.call_args[0][2].shape[0], D.data['Y_test'].shape[0])
        # The model prior to fitting is saved, this cannot be directly tested
        # because of the way the mock module is used. Instead, we test whether
        # the if block in which model assignment is done is accessed
        self.assertTrue(evaluator._added_empty_model)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_partial_cv(self, pipeline_mock):
        D = get_binary_classification_datamanager()
        kfold = StratifiedKFold(y=D.data['Y_train'].flatten(), random_state=1,
                                n_folds=5, shuffle=True)

        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        output_dir = os.path.join(os.getcwd(), '.test_partial_cv')
        D = get_binary_classification_datamanager()
        D.name = 'test'

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(D, backend_api, queue_,
                                   configuration=configuration,
                                   cv=kfold,
                                   with_predictions=True,
                                   all_scoring_functions=False,
                                   output_y_test=True)

        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, None)

        evaluator.partial_fit_predict_and_loss(1)

        duration, result, seed, run_info, status = evaluator.queue.get(timeout=1)
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(evaluator.file_output.call_count, 0)
        self.assertEqual(result, 1.0)
        self.assertEqual(pipeline_mock.fit.call_count, 1)
        self.assertEqual(pipeline_mock.predict_proba.call_count, 3)
        # The model prior to fitting is saved, this cannot be directly tested
        # because of the way the mock module is used. Instead, we test whether
        # the if block in which model assignment is done is accessed
        self.assertTrue(evaluator._added_empty_model)

    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_iterative_partial_cv(self, pipeline_mock):
        # Regular fitting
        D = get_binary_classification_datamanager()
        D.name = 'test'
        kfold = StratifiedKFold(y=D.data['Y_train'].flatten(), random_state=1, n_folds=3)

        class SideEffect(object):
            def __init__(self):
                self.fully_fitted_call_count = 0

            def configuration_fully_fitted(self):
                self.fully_fitted_call_count += 1
                return self.fully_fitted_call_count > 5

        Xt_fixture = 'Xt_fixture'
        pipeline_mock.estimator_supports_iterative_fit.return_value = True
        pipeline_mock.configuration_fully_fitted.side_effect = SideEffect().configuration_fully_fitted
        pipeline_mock.pre_transform.return_value = Xt_fixture, {}
        pipeline_mock.predict_proba.side_effect = lambda X, batch_size: np.tile([0.6, 0.4], (len(X), 1))
        pipeline_mock.side_effect = lambda **kwargs: pipeline_mock
        output_dir = os.path.join(os.getcwd(), '.test_iterative_partial_cv')

        configuration = unittest.mock.Mock(spec=Configuration)
        backend_api = backend.create(output_dir, output_dir)
        queue_ = multiprocessing.Queue()

        evaluator = TrainEvaluator(D, backend_api, queue_,
                                   configuration=configuration,
                                   cv=kfold,
                                   with_predictions=True,
                                   all_scoring_functions=False,
                                   output_y_test=True)
        evaluator.file_output = unittest.mock.Mock(spec=evaluator.file_output)
        evaluator.file_output.return_value = (None, None)

        class LossSideEffect(object):
            def __init__(self):
                self.losses = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
                self.iteration = 0

            def side_effect(self, *args):
                self.iteration += 1
                return self.losses[self.iteration]
        evaluator._loss = unittest.mock.Mock()
        evaluator._loss.side_effect = LossSideEffect().side_effect

        evaluator.partial_fit_predict_and_loss(fold=1, iterative=True)
        # No file output here!
        self.assertEqual(evaluator.file_output.call_count, 0)

        for i in range(1, 6):
            duration, result, seed, run_info, status = evaluator.queue.get(timeout=1)
            self.assertAlmostEqual(result, 1.0 - (0.2 * i))
        self.assertRaises(queue.Empty, evaluator.queue.get, timeout=1)

        self.assertEqual(pipeline_mock.iterative_fit.call_count, 5)
        self.assertEqual([cal[1]['n_iter'] for cal in pipeline_mock.iterative_fit.call_args_list], [2, 4, 8, 16, 32])
        # fifteen calls because of the holdout, the validation and the test set
        # and a total of five calls because of five iterations of fitting
        self.assertFalse(hasattr(evaluator, 'model'))
        self.assertEqual(pipeline_mock.iterative_fit.call_count, 5)
        # fifteen calls because of the holdout, the validation and the test set
        # and a total of five calls because of five iterations of fitting
        self.assertEqual(pipeline_mock.predict_proba.call_count, 15)

    @unittest.mock.patch('autosklearn.util.backend.Backend')
    @unittest.mock.patch('os.makedirs')
    def test_file_output(self, makedirs_mock, backend_mock):

        D = get_regression_datamanager()
        D.name = 'test'
        configuration = unittest.mock.Mock(spec=Configuration)
        queue_ = multiprocessing.Queue()

        kfold = StratifiedKFold(y=D.data['Y_train'].flatten(),
                                n_folds=5, shuffle=True, random_state=1)
        evaluator = TrainEvaluator(D, backend_mock, queue=queue_,
                                   configuration=configuration,
                                   cv=kfold,
                                   with_predictions=True,
                                   all_scoring_functions=True,
                                   output_y_test=True)

        backend_mock.get_model_dir.return_value = True
        evaluator.model = 'model'
        evaluator.Y_optimization = D.data['Y_train']
        rval = evaluator.file_output(D.data['Y_train'], D.data['Y_valid'],
                                     D.data['Y_test'])

        self.assertEqual(rval, (None, None))
        self.assertEqual(backend_mock.save_targets_ensemble.call_count, 1)
        self.assertEqual(backend_mock.save_predictions_as_npy.call_count, 3)
        self.assertEqual(makedirs_mock.call_count, 1)
        self.assertEqual(backend_mock.save_model.call_count, 1)

        # Check for not containing NaNs - that the models don't predict nonsense
        # for unseen data
        D.data['Y_valid'][0] = np.NaN
        rval = evaluator.file_output(D.data['Y_train'], D.data['Y_valid'],
                                     D.data['Y_test'])
        self.assertEqual(rval, (1.0, 'Model predictions for validation set contains NaNs.'))
        D.data['Y_train'][0] = np.NaN
        rval = evaluator.file_output(D.data['Y_train'], D.data['Y_valid'],
                                     D.data['Y_test'])
        self.assertEqual(rval, (1.0, 'Model predictions for optimization set contains NaNs.'))

    @unittest.mock.patch('autosklearn.util.backend.Backend')
    @unittest.mock.patch('autosklearn.pipeline.classification.SimpleClassificationPipeline')
    def test_predict_proba_binary_classification(self, mock, backend_mock):
        D = get_binary_classification_datamanager()
        mock.predict_proba.side_effect = lambda y, batch_size: np.array([[0.1, 0.9]] * 7)
        mock.side_effect = lambda **kwargs: mock

        configuration = unittest.mock.Mock(spec=Configuration)
        queue_ = multiprocessing.Queue()
        kfold = ShuffleSplit(n=len(D.data['Y_train']), random_state=1, n_iter=1)

        evaluator = TrainEvaluator(D, backend_mock, queue_,
                                   configuration=configuration,
                                   cv=kfold)
        evaluator.fit_predict_and_loss()
        Y_optimization_pred = backend_mock.save_predictions_as_npy.call_args_list[0][0][0]
        print(Y_optimization_pred)

        for i in range(7):
            self.assertEqual(0.9, Y_optimization_pred[i][1])

    def test_get_results(self):
        backend_mock = unittest.mock.Mock(spec=backend.Backend)
        backend_mock.get_model_dir.return_value = 'dutirapbdxvltcrpbdlcatepdeau'
        D = get_binary_classification_datamanager()
        kfold = ShuffleSplit(n=len(D.data['Y_train']), random_state=1, n_iter=1)
        queue_ = multiprocessing.Queue()
        for i in range(5):
            queue_.put((i * 1, 1 - (i * 0.2), 0, "", StatusType.SUCCESS))
        result = get_last_result(queue_)
        self.assertEqual(result[0], 4)
        self.assertAlmostEqual(result[1], 0.2)

    def test_datasets(self):
        for getter in get_dataset_getters():
            testname = '%s_%s' % (os.path.basename(__file__).
                                  replace('.pyc', '').replace('.py', ''),
                                  getter.__name__)

            with self.subTest(testname):
                backend_mock = unittest.mock.Mock(spec=backend.Backend)
                backend_mock.get_model_dir.return_value = 'dutirapbdxvltcrpbdlcatepdeau'
                D = getter()
                D_ = copy.deepcopy(D)
                y = D.data['Y_train']
                if len(y.shape) == 2 and y.shape[1] == 1:
                    y = y.flatten()
                kfold = ShuffleSplit(n=len(y), n_iter=5, random_state=1)
                queue_ = multiprocessing.Queue()
                evaluator = TrainEvaluator(D_, backend_mock, queue_,
                                           cv=kfold)

                evaluator.fit_predict_and_loss()
                duration, result, seed, run_info, status = evaluator.queue.get(timeout=1)
                self.assertTrue(np.isfinite(result))


class FunctionsTest(unittest.TestCase):
    def setUp(self):
        self.queue = multiprocessing.Queue()
        self.configuration = get_configuration_space(
            {'task': MULTICLASS_CLASSIFICATION,
             'is_sparse': False}).get_default_configuration()
        self.data = get_multiclass_classification_datamanager()
        self.tmp_dir = os.path.join(os.path.dirname(__file__),
                                    '.test_holdout_functions')
        self.n = len(self.data.data['Y_train'])
        self.y = self.data.data['Y_train'].flatten()
        self.backend = unittest.mock.Mock()
        self.backend.get_model_dir.return_value = 'udiaetzrpduaeirdaetr'
        self.backend.output_directory = 'duapdbaetpdbe'

    def test_eval_holdout(self):
        kfold = ShuffleSplit(n=self.n, random_state=1, n_iter=1, test_size=0.33)
        eval_holdout(self.queue, self.configuration, self.data, self.backend,
                     kfold, 1, 1, None, True, False, True, None, None, False)
        info = get_last_result(self.queue)
        self.assertAlmostEqual(info[1], 0.095, places=3)
        self.assertEqual(info[2], 1)
        self.assertNotIn('bac_metric', info[3])

    def test_eval_holdout_all_loss_functions(self):
        kfold = ShuffleSplit(n=self.n, random_state=1, n_iter=1, test_size=0.33)
        eval_holdout(self.queue, self.configuration, self.data, self.backend,
                     kfold, 1, 1, None, True, True, True, None, None, False)
        info = get_last_result(self.queue)
        self.assertIn('f1_metric: 0.0954545454545;pac_metric: 0.203125867166;'
                      'acc_metric: 0.0909090909091;auc_metric: 0.0197868008145;'
                      'bac_metric: 0.0954545454545;duration: ', info[3])
        self.assertAlmostEqual(info[1], 0.095, places=3)
        self.assertEqual(info[2], 1)

    # def test_eval_holdout_on_subset(self):
    #     backend_api = backend.create(self.tmp_dir, self.tmp_dir)
    #     eval_holdout(self.queue, self.configuration, self.data,
    #                  backend_api, 1, 1, 43, True, False, True, None, None,
    #                  False)
    #     info = self.queue.get()
    #     self.assertAlmostEqual(info[1], 0.1)
    #     self.assertEqual(info[2], 1)

    def test_eval_holdout_iterative_fit_no_timeout(self):
        kfold = ShuffleSplit(n=self.n, random_state=1, n_iter=1, test_size=0.33)
        eval_iterative_holdout(self.queue, self.configuration, self.data,
                               self.backend, kfold, 1, 1, None, True,
                               False, True, None, None, False)
        info = get_last_result(self.queue)
        self.assertAlmostEqual(info[1], 0.09545454545454557)
        self.assertEqual(info[2], 1)

    # def test_eval_holdout_iterative_fit_on_subset_no_timeout(self):
    #     backend_api = backend.create(self.tmp_dir, self.tmp_dir)
    #     eval_iterative_holdout(self.queue, self.configuration,
    #                            self.data, backend_api, 1, 1, 43, True, False,
    #                            True, None, None, False)
    #
    #     info = self.queue.get()
    #     self.assertAlmostEqual(info[1], 0.1)
    #     self.assertEqual(info[2], 1)

    def test_eval_cv(self):
        cv = StratifiedKFold(y=self.y, shuffle=True, random_state=1)
        eval_cv(queue=self.queue, config=self.configuration, data=self.data,
                backend=self.backend, seed=1, num_run=1, cv=cv, subsample=None,
                with_predictions=True, all_scoring_functions=False,
                output_y_test=True, include=None, exclude=None,
                disable_file_output=False)
        info = get_last_result(self.queue)
        self.assertAlmostEqual(info[1], 0.063004032258064502)
        self.assertEqual(info[2], 1)
        self.assertNotIn('bac_metric', info[3])

    def test_eval_cv_all_loss_functions(self):
        cv = StratifiedKFold(y=self.y, shuffle=True, random_state=1)
        eval_cv(queue=self.queue, config=self.configuration, data=self.data,
                backend=self.backend, seed=1, num_run=1, cv=cv, subsample=None,
                with_predictions=True, all_scoring_functions=True,
                output_y_test=True, include=None, exclude=None,
                disable_file_output=False)
        info = get_last_result(self.queue)
        self.assertIn(
            'f1_metric: 0.0635080645161;pac_metric: 0.165226664054;'
            'acc_metric: 0.06;auc_metric: 0.0154405176049;'
            'bac_metric: 0.0630040322581;duration: ', info[3])
        self.assertAlmostEqual(info[1], 0.063004032258064502)
        self.assertEqual(info[2], 1)

    # def test_eval_cv_on_subset(self):
    #     backend_api = backend.create(self.tmp_dir, self.tmp_dir)
    #     eval_cv(queue=self.queue, config=self.configuration, data=self.data,
    #             backend=backend_api, seed=1, num_run=1, folds=5, subsample=45,
    #             with_predictions=True, all_scoring_functions=False,
    #             output_y_test=True, include=None, exclude=None,
    #             disable_file_output=False)
    #     info = self.queue.get()
    #     self.assertAlmostEqual(info[1], 0.063004032258064502)
    #     self.assertEqual(info[2], 1)

    def test_eval_partial_cv(self):
        cv = StratifiedKFold(y=self.y, shuffle=True, random_state=1,
                             n_folds=5)
        results = [0.071428571428571508,
                   0.15476190476190488,
                   0.08333333333333337,
                   0.16666666666666674,
                   0.0]
        for fold in range(5):
            eval_partial_cv(queue=self.queue, config=self.configuration,
                            data=self.data, backend=self.backend, seed=1,
                            num_run=1, instance=fold, cv=cv,
                            subsample=None, with_predictions=True,
                            all_scoring_functions=False, output_y_test=True,
                            include=None, exclude=None,
                            disable_file_output=False)
            info = get_last_result(self.queue)
            results.append(info[1])
            self.assertAlmostEqual(info[1], results[fold])
            self.assertEqual(info[2], 1)

    # def test_eval_partial_cv_on_subset_no_timeout(self):
    #     backend_api = backend.create(self.tmp_dir, self.tmp_dir)
    #
    #     results = [0.071428571428571508,
    #                0.15476190476190488,
    #                0.08333333333333337,
    #                0.16666666666666674,
    #                0.0]
    #     for fold in range(5):
    #         eval_partial_cv(queue=self.queue, config=self.configuration,
    #                         data=self.data, backend=backend_api,
    #                         seed=1, num_run=1, instance=fold, folds=5,
    #                         subsample=80, with_predictions=True,
    #                         all_scoring_functions=False, output_y_test=True,
    #                         include=None, exclude=None,
    #                         disable_file_output=False)
    #
    #         info = self.queue.get()
    #         self.assertAlmostEqual(info[1], results[fold])
    #         self.assertEqual(info[2], 1)
    #
    #     results = [0.071428571428571508,
    #                0.15476190476190488,
    #                0.16666666666666674,
    #                0.0,
    #                0.0]
    #     for fold in range(5):
    #         eval_partial_cv(queue=self.queue, config=self.configuration,
    #                         data=self.data, backend=backend_api,
    #                         seed=1, num_run=1, instance=fold, folds=5,
    #                         subsample=43, with_predictions=True,
    #                         all_scoring_functions=False, output_y_test=True,
    #                         include=None, exclude=None,
    #                         disable_file_output=False)
    #
    #         info = self.queue.get()
    #         self.assertAlmostEqual(info[1], results[fold])
    #         self.assertEqual(info[2], 1)
