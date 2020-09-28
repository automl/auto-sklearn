import json
import os
import random
import shutil
import socket
import subprocess
import unittest

import arff


class TestMetadataGeneration(unittest.TestCase):

    def setUp(self):
        self.working_directory = '/tmp/autosklearn-unittest-tmp-dir-%s-%d-%d' % (
            socket.gethostname(), os.getpid(), random.randint(0, 1000000))

    def print_files(self):
        print('Existing files:')
        for dirpath, dirnames, filenames in os.walk(self.working_directory):
            print(dirpath, dirnames, filenames)

    def test_metadata_generation(self):

        regression_task_id = 5022
        classification_task_id = 75222

        current_directory = __file__
        scripts_directory = os.path.abspath(os.path.join(current_directory,
                                                         '..', '..', '..',
                                                         'scripts'))

        # 1. create working directory
        try:
            os.makedirs(self.working_directory)
        except Exception as e:
            print(e)

        # 2. should be done by the person running the unit tests!

        # 3. create configuration commands
        script_filename = os.path.join(scripts_directory, '01_create_commands.py')
        cmd = 'python3 %s --working-directory %s --test' % (script_filename, self.working_directory)
        rval = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(rval.returncode, 0, msg=str(rval))

        # 4. run one of the commands to get some data
        commands_output_file = os.path.join(self.working_directory, 'metadata_commands.txt')
        self.assertTrue(os.path.exists(commands_output_file))

        with open(commands_output_file) as fh:
            cmds = fh.read().split('\n')
            # 6 regression, 11 classification (roc_auc + task 258 is illegal), 1 empty line
            self.assertEqual(len(cmds), 18)

        for task_id, task_type, metric in (
                (classification_task_id, 'classification', 'accuracy'),
                (regression_task_id, 'regression', 'r2')
        ):
            with open(commands_output_file) as fh:
                while True:
                    cmd = fh.readline()
                    if 'task-id %d' % task_id in cmd:
                        break

            self.assertIn('time-limit 86400', cmd)
            self.assertIn('per-run-time-limit 1800', cmd)
            cmd = cmd.replace('time-limit 86400', 'time-limit 60').replace(
                'per-run-time-limit 1800', 'per-run-time-limit 7')
            # This tells the script to use the same memory limit for testing as
            # for training. In production, it would use twice as much!
            cmd = cmd.replace('-s 1', '-s 1 --unittest')
            print('COMMAND: %s' % cmd)
            rval = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
            print('STDOUT: %s' % repr(rval.stdout), flush=True)
            print('STDERR: %s' % repr(rval.stderr), flush=True)

            self.print_files()

            expected_output_directory = os.path.join(self.working_directory,
                                                     'configuration',
                                                     task_type,
                                                     str(task_id), metric)
            self.assertTrue(os.path.exists(expected_output_directory),
                            msg=expected_output_directory)
            smac_log = os.path.join(expected_output_directory, 'AutoML(1):%d.log' % task_id)
            with open(smac_log) as fh:
                smac_output = fh.read()
            self.assertEqual(rval.returncode, 0, msg=str(rval) + '\n' + smac_output)
            expected_validation_output = os.path.join(expected_output_directory,
                                                      'smac3-output',
                                                      'run_1',
                                                      'validation_trajectory.json')
            self.assertTrue(os.path.exists(expected_validation_output))
            trajectory = os.path.join(expected_output_directory,
                                      'smac3-output', 'run_1', 'trajectory.json')

            with open(expected_validation_output) as fh_validation:
                with open(trajectory) as fh_trajectory:
                    traj = json.load(fh_trajectory)
                    valid_traj = json.load(fh_validation)
                    print('Validation trajectory:')
                    print(valid_traj)
                    self.assertGreater(len(traj), 0)
                    self.assertEqual(len(traj), len(valid_traj))

        # 5. Get the test performance of these configurations
        script_filename = os.path.join(scripts_directory, '02_retrieve_metadata.py')
        cmd = 'python3 %s --working-directory %s ' % (script_filename, self.working_directory)
        print('COMMAND: %s' % cmd)
        rval = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        print('STDOUT: %s' % repr(rval.stdout), flush=True)
        print('STDERR: %s' % repr(rval.stderr), flush=True)
        self.assertEqual(rval.returncode, 0, msg=str(rval))

        for file in ['algorithm_runs.arff', 'configurations.csv', 'description.results.txt']:
            for metric in ['accuracy', 'balanced_accuracy', 'log_loss', 'roc_auc']:
                path = os.path.join(
                    self.working_directory,
                    'configuration_results',
                    '%s_binary.classification_dense' % metric,
                    file,
                )
                self.assertTrue(os.path.exists(path), msg=path)

        for file in ['algorithm_runs.arff', 'configurations.csv', 'description.results.txt']:
            for metric in ['r2', 'mean_squared_error']:
                path = os.path.join(
                    self.working_directory,
                    'configuration_results',
                    '%s_regression_dense' % metric,
                    file,
                )
                self.assertTrue(os.path.exists(path), msg=path)

        # 6. Calculate metafeatures
        script_filename = os.path.join(scripts_directory, '03_calculate_metafeatures.py')
        cmd = (
            'python3 %s --working-directory %s --test-mode '
            % (script_filename, self.working_directory)
        )
        rval = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        self.assertEqual(rval.returncode, 0, msg=str(rval))
        for task_type in ('classification', 'regression'):
            for file in ['calculation_times.csv', 'description.features.txt',
                         'feature_costs.arff', 'feature_runstatus.arff',
                         'feature_values.arff']:
                self.assertTrue(
                    os.path.exists(os.path.join(
                        self.working_directory,
                        'metafeatures',
                        task_type,
                        file)
                    )
                )

        with open(
            os.path.join(
                self.working_directory, 'metafeatures', 'regression', 'feature_values.arff'
            )
        ) as fh:
            metafeatures_arff = fh.read().split('\n')
            contains_regression_id = False
            for line in metafeatures_arff:
                if line.startswith('2280,'):
                    contains_regression_id = True
            self.assertTrue(contains_regression_id, msg=metafeatures_arff)

        with open(
                os.path.join(
                    self.working_directory, 'metafeatures', 'classification', 'feature_values.arff'
                )
        ) as fh:
            metafeatures_arff = fh.read().split('\n')
            contains_classification_id = False
            for line in metafeatures_arff:
                if line.startswith('233,'):
                    contains_classification_id = True
            self.assertTrue(contains_classification_id, msg=metafeatures_arff)

        # 7. Create aslib files
        script_filename = os.path.join(scripts_directory, '04_create_aslib_files.py')
        cmd = 'python3 %s --working-directory %s ' % (
            script_filename, self.working_directory)
        rval = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        self.assertEqual(rval.returncode, 0, msg=str(rval))

        for metric, combination in (
            ('accuracy', 'accuracy_binary.classification_dense'),
            ('r2', 'r2_regression_dense'),
        ):
            for file in ['algorithm_runs.arff', 'configurations.csv',
                         'description.txt', 'feature_costs.arff',
                         'feature_runstatus.arff', 'feature_values.arff',
                         'readme.txt']:
                self.assertTrue(os.path.exists(os.path.join(self.working_directory,
                                                            'metadata',
                                                            combination,
                                                            file)))

            with open(os.path.join(self.working_directory,
                                   'metadata',
                                   combination,
                                   'algorithm_runs.arff')) as fh:
                algorithm_runs = arff.load(fh)
                self.assertEqual(algorithm_runs['attributes'],
                                 [('instance_id', 'STRING'),
                                  ('repetition', 'NUMERIC'),
                                  ('algorithm', 'STRING'),
                                  (metric, 'NUMERIC'),
                                  ('runstatus',
                                   ['ok', 'timeout', 'memout', 'not_applicable',
                                    'crash', 'other'])])
                self.assertEqual(len(algorithm_runs['data']), 1)
                self.assertEqual(len(algorithm_runs['data'][0]), 5)
                self.assertLess(algorithm_runs['data'][0][3], 0.9)
                self.assertEqual(algorithm_runs['data'][0][4], 'ok')

    def tearDown(self):
        for i in range(5):
            try:
                shutil.rmtree(self.working_directory)
            except Exception:
                pass
