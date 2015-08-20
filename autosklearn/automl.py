# -*- encoding: utf-8 -*-
import hashlib
import multiprocessing
import os

import numpy as np

import lockfile
from HPOlibConfigSpace.converters import pcs_parser
from sklearn.base import BaseEstimator

import six.moves.cPickle as pickle
from autosklearn import submit_process
from autosklearn.constants import *
from autosklearn.data import split_data
from autosklearn.data.Xy_data_manager import XyDataManager
from autosklearn.data.competition_data_manager import CompetitionDataManager
from autosklearn.metalearning import metalearning
from autosklearn.models import evaluator, paramsklearn
from autosklearn.util import StopWatch, get_logger


class AutoML(multiprocessing.Process, BaseEstimator):

    def __init__(self, tmp_dir,
                 output_dir,
                 time_left_for_this_task,
                 per_run_time_limit,
                 log_dir=None,
                 initial_configurations_via_metalearning=25,
                 ensemble_size=1,
                 ensemble_nbest=1,
                 seed=1,
                 ml_memory_limit=3000,
                 metadata_directory=None,
                 queue=None,
                 keep_models=True):
        super(AutoML, self).__init__()
        self._tmp_dir = tmp_dir
        self._output_dir = output_dir
        self._time_left_for_this_task = time_left_for_this_task
        self._per_run_time_limit = per_run_time_limit
        self._log_dir = log_dir
        self._initial_configurations_via_metalearning = initial_configurations_via_metalearning
        self._ensemble_size = ensemble_size
        self._ensemble_nbest = ensemble_nbest
        self._seed = seed
        self._ml_memory_limit = ml_memory_limit
        self._metadata_directory = metadata_directory
        self._queue = queue
        self._keep_models = keep_models

        self._basename = None
        self._stopwatch = None
        self._logger = None
        self._ohe = None

    @staticmethod
    def _get_logger(log_dir, basename, seed):
        return get_logger(
            outputdir=log_dir,
            name='AutoML_%s_%d' % (basename, seed))

    @staticmethod
    def _set_auto_seed(seed):
        env_key = 'AUTOSKLEARN_SEED'
        # Set environment variable:
        env_seed = os.environ.get(env_key)
        if env_seed is not None and int(env_seed) != seed:
            raise ValueError('It seems you have already started an instance '
                             'of AutoSklearn in this thread.')
        else:
            os.environ[env_key] = str(seed)

    @staticmethod
    def _save_ensemble_data(x_data, y_data, tmp_dir):
        """
        Split dataset and store Data for the ensemble script
        :param x_data:
        :param y_data:
        :return:
        """
        _, _, _, y_ensemble = split_data.split_data(x_data, y_data)

        filepath = os.path.join(
            tmp_dir, 'true_labels_ensemble.npy')

        lock_path = filepath + '.lock'
        with lockfile.LockFile(lock_path):
            if not os.path.exists(filepath):
                np.save(filepath, y_ensemble)

    def run(self):
        raise NotImplementedError()

    def fit(self, data_x, y, task=MULTICLASS_CLASSIFICATION,
            metric='acc_metric', feat_type=None, dataset_name=None):
        if dataset_name is None:
            m = hashlib.md5()
            m.update(data_x.data)
            dataset_name = m.hexdigest()

        self._basename = dataset_name

        self._stopwatch = StopWatch()
        self._stopwatch.start_task(self._basename)
        self._stopwatch.start_task('LoadData')

        self._logger = self._get_logger(self._log_dir,
                                        self._basename,
                                        self._seed)

        loaded_data_manager = XyDataManager(data_x, y, task=task, metric=metric,
                                            feat_type=feat_type,
                                            dataset_name=dataset_name,
                                            encode_labels=False)

        return self._fit(loaded_data_manager)

    def fit_automl_dataset(self, basename, input_dir):
        # == Creating a data object with data and information about it
        self._basename = basename

        self._stopwatch = StopWatch()
        self._stopwatch.start_task(self._basename)

        self._logger = self._get_logger(self._log_dir,
                                        self._basename,
                                        self._seed)

        self._stopwatch.start_task('LoadData')

        self._logger.debug('======== Reading and converting data ==========')
        # Encoding the labels will be done after the metafeature calculation!
        loaded_data_manager = CompetitionDataManager(
            self._basename, input_dir,
            verbose=True,
            encode_labels=False)
        loaded_data_manager_str = str(loaded_data_manager).split('\n')
        for part in loaded_data_manager_str:
            self._logger.debug(part)

        return self._fit(loaded_data_manager)

    def _fit(self, data_d):
        # TODO: check that data and task definition fit together!

        self.metric_ = data_d.info['metric']
        self.task_ = data_d.info['task']
        self.target_num_ = data_d.info['target_num']

        self._set_auto_seed(self._seed)
        self._save_ensemble_data(data_d.data['X_train'], data_d.data['Y_train'],
                                 self._tmp_dir)

        time_for_load_data = self._stopwatch.wall_elapsed(self._basename)
        time_left_after_reading = max(0, self._time_left_for_this_task - time_for_load_data)
        self._logger.info('Remaining time after reading %s %5.2f sec' %
                          (self._basename, time_left_after_reading))

        self._stopwatch.stop_task('LoadData')

        # == Calculate metafeatures
        self._stopwatch.start_task('CalculateMetafeatures')
        categorical = [True if feat_type.lower() in ['categorical'] else False
                       for feat_type in data_d.feat_type]

        if self._initial_configurations_via_metalearning <= 0:
            ml = None
        elif data_d.info['task'] in \
                [MULTICLASS_CLASSIFICATION, BINARY_CLASSIFICATION]:
            ml = metalearning.MetaLearning()
            self._logger.debug('Start calculating metafeatures for %s' %
                               self._basename)
            ml.calculate_metafeatures_with_labels(data_d.data['X_train'],
                                                  data_d.data['Y_train'],
                                                  categorical=categorical,
                                                  dataset_name=self._basename)
        else:
            ml = None
            self._logger.critical('Metafeatures not calculated')
        self._stopwatch.stop_task('CalculateMetafeatures')
        self._logger.debug(
            'Calculating Metafeatures (categorical attributes) took %5.2f' %
            self._stopwatch.wall_elapsed('CalculateMetafeatures'))

        self._stopwatch.start_task('OneHot')
        data_d.perform1HotEncoding()
        self._ohe = data_d.encoder_
        self._stopwatch.stop_task('OneHot')

        # == Pickle the data manager
        self._stopwatch.start_task('StoreDatamanager')
        data_manager_path = os.path.join(self._tmp_dir,
                                         self._basename + '_Manager.pkl')
        data_manager_lockfile = data_manager_path + '.lock'
        with lockfile.LockFile(data_manager_lockfile):
            if not os.path.exists(data_manager_path):
                pickle.dump(data_d, open(data_manager_path, 'w'), protocol=-1)
                self._logger.debug('Pickled Datamanager at %s' %
                                  data_manager_path)
            else:
                self._logger.debug('Data manager already presend at %s' %
                                  data_manager_path)
        self._stopwatch.stop_task('StoreDatamanager')

        # = Create a searchspace
        self._stopwatch.start_task('CreateConfigSpace')
        configspace_path = os.path.join(self._tmp_dir, 'space.pcs')
        self.configuration_space = paramsklearn.get_configuration_space(
            data_d.info)

        self.configuration_space_created_hook()

        sp_string = pcs_parser.write(self.configuration_space)
        configuration_space_lockfile = configspace_path + '.lock'
        with lockfile.LockFile(configuration_space_lockfile):
            if not os.path.exists(configspace_path):
                with open(configspace_path, 'w') as fh:
                    fh.write(sp_string)
                self._logger.debug('Configuration space written to %s' %
                                  configspace_path)
            else:
                self._logger.debug('Configuration space already present at %s' %
                                  configspace_path)
        self._stopwatch.stop_task('CreateConfigSpace')

        if ml is None:
            initial_configurations = []
        elif data_d.info['task'] in \
                [MULTICLASS_CLASSIFICATION, BINARY_CLASSIFICATION]:
            self._stopwatch.start_task('CalculateMetafeaturesEncoded')
            ml.calculate_metafeatures_encoded_labels(
                X_train=data_d.data['X_train'],
                Y_train=data_d.data['Y_train'],
                categorical=[False] * data_d.data['X_train'].shape[0],
                dataset_name=self._basename)
            self._stopwatch.stop_task('CalculateMetafeaturesEncoded')
            self._logger.debug(
                'Calculating Metafeatures (encoded attributes) took %5.2fsec' %
                self._stopwatch.wall_elapsed('CalculateMetafeaturesEncoded'))

            self._logger.debug(ml._metafeatures_labels.__repr__(verbosity=2))
            self._logger.debug(
                ml._metafeatures_encoded_labels.__repr__(verbosity=2))

            self._stopwatch.start_task('InitialConfigurations')
            try:
                initial_configurations = ml.create_metalearning_string_for_smac_call(
                    self.configuration_space, self._basename, self.metric_,
                    self.task_,
                    True if data_d.info['is_sparse'] == 1 else False,
                    self._initial_configurations_via_metalearning,
                    self._metadata_directory)
            except Exception as e:
                import traceback

                self._logger.error(str(e))
                self._logger.error(traceback.format_exc())
                initial_configurations = []

            self._stopwatch.stop_task('InitialConfigurations')

            self._logger.debug('Initial Configurations: (%d)',
                              len(initial_configurations))
            for initial_configuration in initial_configurations:
                self._logger.debug(initial_configuration)
            self._logger.debug(
                'Looking for initial configurations took %5.2fsec' %
                self._stopwatch.wall_elapsed('InitialConfigurations'))
            self._logger.info(
                'Time left for %s after finding initial configurations: %5.2fsec'
                % (self._basename, self._time_left_for_this_task -
                   self._stopwatch.wall_elapsed(self._basename)))
        else:
            initial_configurations = []
            self._logger.critical('Metafeatures encoded not calculated')

        # == Set up a directory where all the trained models will be pickled to
        if self._keep_models:
            self.model_directory_ = os.path.join(self._tmp_dir,
                                                 'models_%d' % self._seed)
            os.mkdir(self.model_directory_)
        self.ensemble_indices_directory_ = os.path.join(
            self._tmp_dir, 'ensemble_indices_%d' % self._seed)
        os.mkdir(self.ensemble_indices_directory_)

        # == RUN SMAC
        self._stopwatch.start_task('runSmac')
        # = Create an empty instance file
        instance_file = os.path.join(self._tmp_dir, 'instances.txt')
        instance_file_lock = instance_file + '.lock'
        with lockfile.LockFile(instance_file_lock):
            if not os.path.exists(instance_file_lock):
                with open(instance_file, 'w') as fh:
                    fh.write('holdout')
                self._logger.debug('Created instance file %s' % instance_file)
            else:
                self._logger.debug('Instance file already present at %s' %
                                  instance_file)

        # = Start SMAC
        time_left_for_smac = max(
            0, self._time_left_for_this_task -
               (self._stopwatch.wall_elapsed(self._basename)))
        self._logger.debug('Start SMAC with %5.2fsec time left' %
                          time_left_for_smac)
        proc_smac, smac_call = \
            submit_process.run_smac(dataset_name=self._basename,
                                    dataset=data_manager_path,
                                    tmp_dir=self._tmp_dir,
                                    searchspace=configspace_path,
                                    instance_file=instance_file,
                                    limit=time_left_for_smac,
                                    cutoff_time=self._per_run_time_limit,
                                    initial_challengers=initial_configurations,
                                    memory_limit=self._ml_memory_limit,
                                    seed=self._seed)
        self._logger.debug(smac_call)
        self._stopwatch.stop_task('runSmac')

        # == RUN ensemble builder
        self._stopwatch.start_task('runEnsemble')
        time_left_for_ensembles = max(
            0, self._time_left_for_this_task -
               (self._stopwatch.wall_elapsed(self._basename)))
        self._logger.debug('Start Ensemble with %5.2fsec time left' %
                          time_left_for_ensembles)
        proc_ensembles = \
            submit_process.run_ensemble_builder(tmp_dir=self._tmp_dir,
                                                dataset_name=self._basename,
                                                task_type=self.task_,
                                                metric=self.metric_,
                                                limit=time_left_for_ensembles,
                                                output_dir=self._output_dir,
                                                ensemble_size=self._ensemble_size,
                                                ensemble_nbest=self._ensemble_nbest,
                                                seed=self._seed,
                                                ensemble_indices_output_dir=self.ensemble_indices_directory_)
        self._stopwatch.stop_task('runEnsemble')

        del data_d

        if self._queue is not None:
            self._queue.put([time_for_load_data, data_manager_path,
                            proc_smac, proc_ensembles])
        else:
            proc_smac.wait()
            proc_ensembles.wait()

        # Delete AutoSklearn environment variable
        del os.environ['AUTOSKLEARN_SEED']
        return self

    def predict(self, data_x):
        if self._keep_models is not True:
            raise ValueError(
                "Predict can only be called if 'keep_models==True'")

        model_files = os.listdir(self.model_directory_)
        models = []
        for model_file in model_files:
            model_file = os.path.join(self.model_directory_, model_file)
            with open(model_file) as fh:
                models.append(pickle.load(fh))

        if len(models) == 0:
            raise ValueError('No models fitted!')

        if self._ohe is not None:
            data_x = self._ohe._transform(data_x)

        indices_files = sorted(os.listdir(self.ensemble_indices_directory_))
        indices_file = os.path.join(self.ensemble_indices_directory_,
                                    indices_files[-1])
        with open(indices_file) as fh:
            ensemble_members_run_numbers = pickle.load(fh)

        predictions = []
        for model, model_file in zip(models, model_files):
            num_run = int(model_file.split('.')[0])

            if num_run not in ensemble_members_run_numbers:
                continue

            weight = ensemble_members_run_numbers[num_run]

            X_ = data_x.copy()
            if self.task_ in REGRESSION_TASKS:
                prediction = model.predict(X_)
            else:
                prediction = model.predict_proba(X_)
            predictions.append(prediction * weight)

        predictions = np.sum(np.array(predictions), axis=0)
        return predictions

    def score(self, data_x, y):
        prediction = self.predict(data_x)
        return evaluator.calculate_score(y, prediction, self.task_,
                                         self.metric_, self.target_num_)

    def configuration_space_created_hook(self):
        pass

    def get_params(self, deep=True):
        raise NotImplementedError('auto-sklearn does not implement '
                                  'get_params() because it is not intended to '
                                  'be optimized.')

    def set_params(self, deep=True):
        raise NotImplementedError('auto-sklearn does not implement '
                                  'set_params() because it is not intended to '
                                  'be optimized.')
