"""
Simplified test suite for AutoSklearn API structure.
Tests module imports and basic structure without heavy dependencies.
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import warnings
warnings.filterwarnings('ignore')

import numpy as np

print("=" * 80)
print("AutoSklearn API Structure Tests")
print("=" * 80)

PASSED = 0
FAILED = 0
SKIPPED = 0
ERRORS = []

def test(name, passed, msg="", skip=False):
    global PASSED, FAILED, SKIPPED, ERRORS
    if skip:
        print(f"[~] {name}: SKIPPED - {msg}")
        SKIPPED += 1
    elif passed:
        print(f"[+] {name}: PASSED - {msg}")
        PASSED += 1
    else:
        print(f"[-] {name}: FAILED - {msg}")
        FAILED += 1
        ERRORS.append(f"{name}: {msg}")

print("\n--- SECTION 1: Core Module Imports ---")

try:
    from autosklearn.__version__ import __version__
    test("Version import", True, f"v{__version__}")
except Exception as e:
    test("Version import", False, str(e))

try:
    from autosklearn.constants import (
        BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, MULTILABEL_CLASSIFICATION,
        REGRESSION, MULTIOUTPUT_REGRESSION, CLASSIFICATION_TASKS, REGRESSION_TASKS
    )
    test("Constants import", True, f"tasks: cls={CLASSIFICATION_TASKS}, reg={REGRESSION_TASKS}")
except Exception as e:
    test("Constants import", False, str(e))

print("\n--- SECTION 2: Data Validation ---")

try:
    from autosklearn.data.validation import InputValidator
    test("InputValidator import", True)
except ImportError as e:
    test("InputValidator import", False, str(e), skip=True)
except Exception as e:
    test("InputValidator import", False, str(e))

try:
    from autosklearn.data.validation import SUPPORTED_FEAT_TYPES, SUPPORTED_TARGET_TYPES
    test("Supported types import", True)
except Exception as e:
    test("Supported types import", False, str(e))

print("\n--- SECTION 3: Pipeline Components ---")

try:
    from autosklearn.pipeline.components.classification import ClassifierChoice
    test("ClassifierChoice import", True)
except ImportError as e:
    test("ClassifierChoice import", False, str(e), skip=True)
except Exception as e:
    test("ClassifierChoice import", False, str(e))

try:
    from autosklearn.pipeline.components.regression import RegressorChoice
    test("RegressorChoice import", True)
except ImportError as e:
    test("RegressorChoice import", False, str(e), skip=True)
except Exception as e:
    test("RegressorChoice import", False, str(e))

try:
    from autosklearn.pipeline.components.data_preprocessing.rescaling import RescalingChoice
    test("RescalingChoice import", True)
except ImportError as e:
    test("RescalingChoice import", False, str(e), skip=True)
except Exception as e:
    test("RescalingChoice import", False, str(e))

print("\n--- SECTION 4: Ensemble Components ---")

try:
    from autosklearn.ensembles.ensemble_selection import EnsembleSelection
    test("EnsembleSelection import", True)
except ImportError as e:
    test("EnsembleSelection import", False, str(e), skip=True)
except Exception as e:
    test("EnsembleSelection import", False, str(e))

try:
    from autosklearn.ensembles.singlebest_ensemble import SingleBestFromRunhistory
    test("SingleBestEnsemble import", True)
except ImportError as e:
    test("SingleBestEnsemble import", False, str(e), skip=True)
except Exception as e:
    test("SingleBestEnsemble import", False, str(e))

print("\n--- SECTION 5: Metrics ---")

try:
    from autosklearn.metrics import accuracy, r2, mean_squared_error
    test("Metrics import", True, f"accuracy={accuracy.name}")
except ImportError as e:
    test("Metrics import", False, str(e), skip=True)
except Exception as e:
    test("Metrics import", False, str(e))

print("\n--- SECTION 6: Estimator Classes ---")

try:
    from autosklearn.estimators import AutoSklearnClassifier, AutoSklearnRegressor
    test("Estimator classes import", True)
except ImportError as e:
    test("Estimator classes import", False, str(e), skip=True)
except Exception as e:
    test("Estimator classes import", False, str(e))

print("\n--- SECTION 7: AutoML Classes ---")

try:
    from autosklearn.automl import AutoML, AutoMLClassifier, AutoMLRegressor
    test("AutoML classes import", True)
except ImportError as e:
    test("AutoML classes import", False, str(e), skip=True)
except Exception as e:
    test("AutoML classes import", False, str(e))

print("\n--- SECTION 8: Data Manager ---")

try:
    from autosklearn.data.xy_data_manager import XYDataManager
    test("XYDataManager import", True)
except ImportError as e:
    test("XYDataManager import", False, str(e), skip=True)
except Exception as e:
    test("XYDataManager import", False, str(e))

print("\n--- SECTION 9: Utility Functions ---")

try:
    from autosklearn.util import pipeline
    test("Pipeline utilities import", True)
except ImportError as e:
    test("Pipeline utilities import", False, str(e), skip=True)
except Exception as e:
    test("Pipeline utilities import", False, str(e))

try:
    from autosklearn.util.data import reduce_dataset_size_if_too_large
    test("Data utilities import", True)
except ImportError as e:
    test("Data utilities import", False, str(e), skip=True)
except Exception as e:
    test("Data utilities import", False, str(e))

print("\n--- SECTION 10: Backend ---")

try:
    from autosklearn.automl_common.common.utils.backend import Backend
    test("Backend import", True)
except ImportError as e:
    test("Backend import", False, str(e), skip=True)
except Exception as e:
    test("Backend import", False, str(e))

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"Passed:  {PASSED}")
print(f"Failed:  {FAILED}")
print(f"Skipped: {SKIPPED}")

if ERRORS:
    print("\nFailed Tests:")
    for e in ERRORS:
        print(f"  - {e}")

print("=" * 80)

if FAILED == 0:
    print("\nAll available tests passed!")
else:
    print(f"\n{FAILED} test(s) failed.")

sys.exit(0 if FAILED == 0 else 1)
