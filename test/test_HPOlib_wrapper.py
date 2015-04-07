import os
import subprocess
import unittest


class HPOlib_wrapperTest(unittest.TestCase):
    """Test the HPOlib wrapper. Do this via the command line/subprocess
    module, to have this call HPOlib_wrapper.py like it is called from the
    HPOlib.
    """
    # TODO also the the wrapper programmatic, so far this rather tests the
    # single evaluators!
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), ".data")
        self.dataset = "31_bac"
        self.param_string = " --params " \
                            "-classifier random_forest " \
                            "-imputation:strategy mean " \
                            "-preprocessor no_preprocessing " \
                            "-random_forest:bootstrap True " \
                            "-random_forest:criterion gini " \
                            "-random_forest:max_depth None " \
                            "-random_forest:max_features 1.0 " \
                            "-random_forest:max_leaf_nodes None " \
                            "-random_forest:min_samples_leaf 1 " \
                            "-random_forest:min_samples_split 2 " \
                            "-random_forest:n_estimators 100 " \
                            "-rescaling:strategy min/max"

    def test_holdout(self):
        call = "python -m autosklearn.HPOlib_wrapper --dataset %s " \
               "--data_dir %s --fold 0 --folds 1 --seed 1 %s" % \
               (self.dataset, self.data_dir, self.param_string)
        proc = subprocess.Popen(call, shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        proc.wait()
        self.assertEqual(proc.stderr.read(), "")
        output = proc.stdout.read().split(",")
        result = float(output[3])
        additional = output[5]
        self.assertAlmostEqual(result, 0.740202)
        # Metrics in the additional data are seperated by a semicolon. Right
        # now, we have five different metrics plus duration
        # holdout has an additional num_run field
        self.assertEqual(additional.count(";"), 6)

    def test_testset(self):
        call = "python -m autosklearn.HPOlib_wrapper --dataset %s " \
               "--data_dir %s --fold 0 --folds 1 --test True --seed 1 %s" % \
               (self.dataset, self.data_dir, self.param_string)
        proc = subprocess.Popen(call, shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        proc.wait()
        self.assertEqual(proc.stderr.read(), "")
        output = proc.stdout.read().split(",")
        result = float(output[3])
        additional = output[5]
        self.assertAlmostEqual(0.670996, result)
        self.assertEqual(additional.count(";"), 5)

    def test_cv(self):
        call = "python -m autosklearn.HPOlib_wrapper --dataset %s " \
               "--data_dir %s --fold 0 --folds 1 --cv 3 --seed 1 %s" % \
               (self.dataset, self.data_dir, self.param_string)
        proc = subprocess.Popen(call, shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        proc.wait()
        self.assertEqual(proc.stderr.read(), "")
        output = proc.stdout.read().split(",")
        result = float(output[3])
        additional = output[5]
        # Has num_run in the additional info
        self.assertEqual(additional.count(";"), 6)
        self.assertEqual(0.756219, result)

    def test_partial_cv(self):
        results = []
        for fold in range(3):
            call = "python -m autosklearn.HPOlib_wrapper --dataset %s " \
                   "--data_dir %s --fold %d --folds 3 --cv 3 --seed 1 %s" % \
                   (self.dataset, self.data_dir, fold, self.param_string)
            proc = subprocess.Popen(call, shell=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            proc.wait()
            self.assertEqual(proc.stderr.read(), "")
            output = proc.stdout.read().split(",")
            result = float(output[3])
            additional = output[5]
            results.append(result)
            self.assertEqual(additional.count(";"), 5)
        self.assertEqual([0.801217, 0.742155, 0.725029], results)

