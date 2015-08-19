# -*- encoding: utf-8 -*-
import os
import subprocess
import unittest


class HPOlib_interfaceTest(unittest.TestCase):

    """Test the HPOlib wrapper.

    Do this via the command line/subprocess module, to have this call
    HPOlib_wrapper.py like it is called from the HPOlib.

    """

    # TODO also the the wrapper programmatic, so far this rather tests the
    # single evaluators!
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '../.data')
        self.dataset = '31_bac'
        self.dataset_string = os.path.join(self.data_dir, self.dataset)
        self.param_string = ' --params ' \
                            '-balancing:strategy none ' \
                            '-classifier random_forest ' \
                            '-imputation:strategy mean ' \
                            '-preprocessor no_preprocessing ' \
                            '-random_forest:bootstrap True ' \
                            '-random_forest:criterion gini ' \
                            '-random_forest:max_depth None ' \
                            '-random_forest:max_features 1.0 ' \
                            '-random_forest:max_leaf_nodes None ' \
                            '-random_forest:min_samples_leaf 1 ' \
                            '-random_forest:min_samples_split 2 ' \
                            '-random_forest:n_estimators 100 ' \
                            '-rescaling:strategy min/max'

    def tearDown(self):
        try:
            manager = os.path.join(os.path.dirname(__file__),
                                   '%s_Manager.pkl' % self.dataset)
            os.remove(manager)
        except Exception:
            pass

    def test_holdout(self):
        call = 'python -m autosklearn.cli.HPOlib_interface --dataset %s ' \
               '--fold 0 --folds 1 --seed 1 --mode holdout %s' % \
               (self.dataset_string, self.param_string)
        proc = subprocess.Popen(call,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        proc.wait()
        self.assertEqual(proc.stderr.read(), '')
        output = proc.stdout.read().split(',')
        try:
            result = float(output[3])
        except Exception:
            pass
        additional = output[5]
        self.assertAlmostEqual(result, 0.740202)
        # Metrics in the additional data are seperated by a semicolon. Right
        # now, we have five different metrics plus duration
        # holdout has an additional num_run field
        self.assertEqual(additional.count(';'), 6)

    def test_testset(self):
        call = 'python -m autosklearn.cli.HPOlib_interface --dataset %s ' \
               '--fold 0 --folds 1 --test True --seed 1 ' \
               '--mode test %s' % \
               (self.dataset_string, self.param_string)
        proc = subprocess.Popen(call,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        proc.wait()
        self.assertEqual(proc.stderr.read(), '')
        output = proc.stdout.read().split(',')
        try:
            result = float(output[3])
        except Exception:
            print(output)
        additional = output[5]
        self.assertAlmostEqual(0.670996, result)
        self.assertEqual(additional.count(';'), 5)

    def test_cv(self):
        call = 'python -m autosklearn.cli.HPOlib_interface --dataset %s ' \
               '--fold 0 --folds 1 --seed 1 ' \
               '--mode 3cv %s' % \
               (self.dataset_string, self.param_string)
        proc = subprocess.Popen(call,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        proc.wait()
        self.assertEqual(proc.stderr.read(), '')
        output = proc.stdout.read().split(',')
        try:
            result = float(output[3])
        except Exception:
            print(output)
        additional = output[5]
        # Has num_run in the additional info
        self.assertEqual(additional.count(';'), 6)
        self.assertEqual(0.779673, result)

    def test_partial_cv(self):
        results = []
        for fold in range(3):
            call = 'python -m autosklearn.cli.HPOlib_interface --dataset %s ' \
                   '--fold %d --folds 3 --mode cv --seed 1 ' \
                   '%s' % \
                   (self.dataset_string, fold, self.param_string)
            proc = subprocess.Popen(call,
                                    shell=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            proc.wait()
            self.assertEqual(proc.stderr.read(), '')
            output = proc.stdout.read().split(',')
            try:
                result = float(output[3])
            except Exception:
                print(output)
            additional = output[5]
            results.append(result)
            self.assertEqual(additional.count(';'), 5)
        self.assertEqual([0.795038, 0.827497, 0.716609], results)

    def test_nested_cv(self):
        call = 'python -m autosklearn.cli.HPOlib_interface --dataset %s ' \
               '--fold 0 --folds 1 --seed 1 ' \
               '--mode 3/3-nested-cv %s' % \
               (self.dataset_string, self.param_string)
        proc = subprocess.Popen(call,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        proc.wait()
        self.assertEqual(proc.stderr.read(), '')
        output = proc.stdout.read().split(',')
        try:
            result = float(output[3])
        except Exception:
            print(output)
        additional = output[5]
        # Has num_run in the additional info
        self.assertEqual(additional.count(';'), 11)
        self.assertEqual(0.815061, result)
