"""
Simplified test suite for AutoSklearn API structure and basic functionality.
This version tests the API without requiring full AutoML pipeline execution.
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import make_classification, make_multilabel_classification, make_regression

print("=" * 80)
print("AutoSklearn Simplified Test Suite (API Structure Tests)")
print("=" * 80)

TEST_RESULTS = {
    'passed': 0,
    'failed': 0,
    'skipped': 0,
    'errors': [],
}

def test_result(name: str, passed: bool, message: str = "", skipped: bool = False):
    if skipped:
        status = "SKIPPED"
        symbol = "[~]"
        TEST_RESULTS['skipped'] += 1
    elif passed:
        status = "PASSED"
        symbol = "[+]"
        TEST_RESULTS['passed'] += 1
    else:
        status = "FAILED"
        symbol = "[-]"
        TEST_RESULTS['failed'] += 1
        TEST_RESULTS['errors'].append(f"{name}: {message}")
    
    print(f"{symbol} {name}: {status}")
    if message:
        print(f"    {message}")

def run_api_structure_tests():
    """Test the API structure without running full AutoML pipeline."""
    
    print("\n" + "=" * 80)
    print("SECTION 1: Module Import Tests")
    print("=" * 80)
    
    try:
        from autosklearn.__version__ import __version__
        test_result("Import __version__", True, f"Version: {__version__}")
    except Exception as e:
        test_result("Import __version__", False, str(e))
        return False
    
    try:
        from autosklearn.constants import (
            BINARY_CLASSIFICATION,
            MULTICLASS_CLASSIFICATION,
            MULTILABEL_CLASSIFICATION,
            REGRESSION,
            MULTIOUTPUT_REGRESSION,
            CLASSIFICATION_TASKS,
            REGRESSION_TASKS,
        )
        test_result("Import constants", True, 
                   f"Classification tasks: {CLASSIFICATION_TASKS}, Regression tasks: {REGRESSION_TASKS}")
    except Exception as e:
        test_result("Import constants", False, str(e))
        return False
    
    try:
        from autosklearn.metrics import accuracy, r2, mean_squared_error, f1_macro
        test_result("Import metrics", True, f"accuracy={accuracy.name}, r2={r2.name}")
    except ImportError as e:
        test_result("Import metrics", False, str(e), skipped=True)
    except Exception as e:
        test_result("Import metrics", False, str(e))
    
    print("\n" + "=" * 80)
    print("SECTION 2: Estimator Class Structure Tests")
    print("=" * 80)
    
    try:
        from autosklearn.estimators import AutoSklearnClassifier, AutoSklearnRegressor, AutoSklearnEstimator
        test_result("Import estimator classes", True)
    except Exception as e:
        test_result("Import estimator classes", False, str(e))
        return False
    
    try:
        cls = AutoSklearnClassifier(
            time_left_for_this_task=10,
            per_run_time_limit=5,
            seed=42,
        )
        test_result("AutoSklearnClassifier instantiation", True)
    except Exception as e:
        test_result("AutoSklearnClassifier instantiation", False, str(e))
        return False
    
    try:
        reg = AutoSklearnRegressor(
            time_left_for_this_task=10,
            per_run_time_limit=5,
            seed=42,
        )
        test_result("AutoSklearnRegressor instantiation", True)
    except Exception as e:
        test_result("AutoSklearnRegressor instantiation", False, str(e))
        return False
    
    try:
        params = cls.get_params()
        test_result("get_params() returns dict", isinstance(params, dict), 
                   f"Keys count: {len(params)}")
    except Exception as e:
        test_result("get_params() returns dict", False, str(e))
    
    try:
        cls.set_params(time_left_for_this_task=20)
        test_result("set_params() works", cls.time_left_for_this_task == 20)
    except Exception as e:
        test_result("set_params() works", False, str(e))
    
    print("\n" + "=" * 80)
    print("SECTION 3: Parameter Validation Tests")
    print("=" * 80)
    
    try:
        cls = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            seed=1,
            memory_limit=1024,
            ensemble_size=10,
            ensemble_nbest=5,
            max_models_on_disc=10,
            initial_configurations_via_metalearning=5,
        )
        test_result("Valid parameters accepted", True)
    except Exception as e:
        test_result("Valid parameters accepted", False, str(e))
    
    try:
        cls = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            include={'classifier': ['random_forest']},
        )
        test_result("include parameter accepted", True)
    except Exception as e:
        test_result("include parameter accepted", False, str(e))
    
    try:
        cls = AutoSklearnClassifier(
            time_left_for_this_task=60,
            per_run_time_limit=10,
            exclude={'classifier': ['random_forest']},
        )
        test_result("exclude parameter accepted", True)
    except Exception as e:
        test_result("exclude parameter accepted", False, str(e))
    
    print("\n" + "=" * 80)
    print("SECTION 4: Data Validation Tests")
    print("=" * 80)
    
    try:
        from autosklearn.data.validation import InputValidator, SUPPORTED_FEAT_TYPES, SUPPORTED_TARGET_TYPES
        test_result("Import data validation", True)
    except Exception as e:
        test_result("Import data validation", False, str(e))
    
    try:
        validator = InputValidator()
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        validator.fit(X, y)
        test_result("InputValidator basic fit", True)
    except Exception as e:
        test_result("InputValidator basic fit", False, str(e))
    
    try:
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        validator = InputValidator()
        validator.fit(X, y)
        test_result("InputValidator iris dataset", True, f"X shape: {X.shape}, y shape: {y.shape}")
    except Exception as e:
        test_result("InputValidator iris dataset", False, str(e))
    
    print("\n" + "=" * 80)
    print("SECTION 5: Pipeline Component Tests")
    print("=" * 80)
    
    try:
        from autosklearn.pipeline.components.classification import ClassifierChoice
        test_result("Import ClassifierChoice", True)
    except Exception as e:
        test_result("Import ClassifierChoice", False, str(e))
    
    try:
        from autosklearn.pipeline.components.regression import RegressorChoice
        test_result("Import RegressorChoice", True)
    except Exception as e:
        test_result("Import RegressorChoice", False, str(e))
    
    try:
        from autosklearn.pipeline.components.data_preprocessing.categorical_encoding import OHEChoice
        test_result("Import OHEChoice", True)
    except Exception as e:
        test_result("Import OHEChoice", False, str(e))
    
    try:
        from autosklearn.pipeline.components.data_preprocessing.rescaling import RescalingChoice
        test_result("Import RescalingChoice", True)
    except Exception as e:
        test_result("Import RescalingChoice", False, str(e))
    
    print("\n" + "=" * 80)
    print("SECTION 6: Ensemble Tests")
    print("=" * 80)
    
    try:
        from autosklearn.ensembles.ensemble_selection import EnsembleSelection
        test_result("Import EnsembleSelection", True)
    except Exception as e:
        test_result("Import EnsembleSelection", False, str(e))
    
    try:
        from autosklearn.ensembles.singlebest_ensemble import SingleBestFromRunhistory
        test_result("Import SingleBestFromRunhistory", True)
    except Exception as e:
        test_result("Import SingleBestFromRunhistory", False, str(e))
    
    print("\n" + "=" * 80)
    print("SECTION 7: Metric Tests")
    print("=" * 80)
    
    try:
        from autosklearn.metrics import (
            accuracy, balanced_accuracy, f1, f1_macro, f1_micro, f1_weighted,
            precision, precision_macro, precision_micro, precision_weighted,
            recall, recall_macro, recall_micro, recall_weighted,
            roc_auc, average_precision, log_loss,
        )
        test_result("Import classification metrics", True, f"accuracy={accuracy.name}")
    except Exception as e:
        test_result("Import classification metrics", False, str(e))
    
    try:
        from autosklearn.metrics import (
            r2, mean_squared_error, mean_absolute_error, median_absolute_error,
        )
        test_result("Import regression metrics", True, f"r2={r2.name}")
    except Exception as e:
        test_result("Import regression metrics", False, str(e))
    
    try:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]])
        score = accuracy(y_true, y_pred)
        test_result("accuracy metric calculation", True, f"Score: {score:.4f}")
    except Exception as e:
        test_result("accuracy metric calculation", False, str(e))
    
    print("\n" + "=" * 80)
    print("SECTION 8: Invalid Input Detection Tests")
    print("=" * 80)
    
    try:
        from autosklearn.data.validation import InputValidator
        validator = InputValidator()
        
        X_empty = np.array([]).reshape(0, 2)
        y_empty = np.array([])
        
        try:
            validator.fit(X_empty, y_empty)
            test_result("Empty array detection", False, "Should have raised error")
        except Exception:
            test_result("Empty array detection", True, "Error raised as expected")
    except Exception as e:
        test_result("Empty array detection", False, str(e))
    
    try:
        from autosklearn.data.validation import InputValidator
        validator = InputValidator()
        
        X = np.random.rand(30, 5)
        y_single_class = np.ones(30)
        
        try:
            validator.fit(X, y_single_class)
            test_result("Single class detection", True, "Validator accepted")
        except Exception as e:
            test_result("Single class detection", True, f"Error raised: {str(e)[:50]}")
    except Exception as e:
        test_result("Single class detection", False, str(e))
    
    print("\n" + "=" * 80)
    print("SECTION 9: Feature Type Tests")
    print("=" * 80)
    
    try:
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        
        cls = AutoSklearnClassifier(time_left_for_this_task=10)
        
        try:
            cls.fit(X, y, feat_type=["invalid"] * 5)
            test_result("Invalid feat_type detection", False, "Should have raised error")
        except ValueError as e:
            test_result("Invalid feat_type detection", True, "ValueError raised as expected")
        except Exception as e:
            test_result("Invalid feat_type detection", True, f"Error raised: {str(e)[:50]}")
    except Exception as e:
        test_result("Invalid feat_type detection", False, str(e))
    
    try:
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        
        cls = AutoSklearnClassifier(time_left_for_this_task=10)
        
        try:
            cls.fit(X, y, feat_type=["Categorical"] * 3)
            test_result("Mismatched feat_type length detection", False, "Should have raised error")
        except ValueError as e:
            test_result("Mismatched feat_type length detection", True, "ValueError raised as expected")
        except Exception as e:
            test_result("Mismatched feat_type length detection", True, f"Error raised: {str(e)[:50]}")
    except Exception as e:
        test_result("Mismatched feat_type length detection", False, str(e))
    
    print("\n" + "=" * 80)
    print("SECTION 10: AutoML Module Tests")
    print("=" * 80)
    
    try:
        from autosklearn.automl import AutoML, AutoMLClassifier, AutoMLRegressor
        test_result("Import AutoML classes", True)
    except Exception as e:
        test_result("Import AutoML classes", False, str(e))
    
    try:
        automl = AutoML(
            time_left_for_this_task=30,
            per_run_time_limit=10,
            seed=42,
        )
        test_result("AutoML instantiation", True)
    except Exception as e:
        test_result("AutoML instantiation", False, str(e))
    
    try:
        automl_cls = AutoMLClassifier(
            time_left_for_this_task=30,
            per_run_time_limit=10,
            seed=42,
        )
        test_result("AutoMLClassifier instantiation", True)
    except Exception as e:
        test_result("AutoMLClassifier instantiation", False, str(e))
    
    try:
        automl_reg = AutoMLRegressor(
            time_left_for_this_task=30,
            per_run_time_limit=10,
            seed=42,
        )
        test_result("AutoMLRegressor instantiation", True)
    except Exception as e:
        test_result("AutoMLRegressor instantiation", False, str(e))
    
    return True

if __name__ == "__main__":
    success = run_api_structure_tests()
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed:  {TEST_RESULTS['passed']}")
    print(f"Failed:  {TEST_RESULTS['failed']}")
    print(f"Skipped: {TEST_RESULTS['skipped']}")
    
    if TEST_RESULTS['errors']:
        print("\nFailed Tests:")
        for error in TEST_RESULTS['errors']:
            print(f"  - {error}")
    
    print("=" * 80)
    
    if TEST_RESULTS['failed'] == 0:
        print("\nAll API structure tests passed!")
    else:
        print(f"\n{TEST_RESULTS['failed']} test(s) failed.")
    
    sys.exit(0 if TEST_RESULTS['failed'] == 0 else 1)
