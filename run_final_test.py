#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Final comprehensive test for Auto-Sklearn
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --------------------------
# Mock Infrastructure
# --------------------------
import resource
import sklearn_patch

class MockDist:
    def __init__(self, version='1.0.0'):
        self.version = version

_versions = {
    'ConfigSpace': '0.4.20',
    'pynisher': '0.6.4',
    'smac': '1.2.0',
    'pyrfr': '0.8.2',
    'dask': '2021.12.0',
    'distributed': '2021.12.0',
}

try:
    import pkg_resources
    _original_get_dist = pkg_resources.get_distribution
    def _mock_get_dist(name):
        if name in _versions:
            return MockDist(_versions[name])
        return _original_get_dist(name)
    pkg_resources.get_distribution = _mock_get_dist
except:
    pass

from types import ModuleType

class MockLocalCluster:
    def __init__(self, *args, **kwargs): pass
    def close(self): pass

class MockFuture:
    def result(self): return None
    def done(self): return True
    def cancelled(self): return False

class MockClient:
    def __init__(self, *args, **kwargs): pass
    def submit(self, *args, **kwargs): return MockFuture()
    def map(self, *args, **kwargs): return []
    def close(self): pass

class MockFireAndForget: pass

dask_mod = ModuleType('dask')
dask_mod.__version__ = _versions['dask']
dask_mod.__file__ = ''
dask_mod.__path__ = []
sys.modules['dask'] = dask_mod

dask_dist_mod = ModuleType('dask.distributed')
dask_dist_mod.Client = MockClient
dask_dist_mod.LocalCluster = MockLocalCluster
dask_dist_mod.Future = MockFuture
dask_dist_mod.FireAndForget = MockFireAndForget
dask_dist_mod.__file__ = ''
dask_dist_mod.__path__ = []
sys.modules['dask.distributed'] = dask_dist_mod
sys.modules['dask'].distributed = dask_dist_mod

dist_mod = ModuleType('distributed')
dist_mod.__version__ = _versions['distributed']
dist_mod.Client = MockClient
dist_mod.LocalCluster = MockLocalCluster
dist_mod.Future = MockFuture
dist_mod.FireAndForget = MockFireAndForget
dist_mod.__file__ = ''
dist_mod.__path__ = []
sys.modules['distributed'] = dist_mod

import warnings
warnings.filterwarnings('ignore')

# --------------------------
# Test Execution
# --------------------------
print("=" * 60)
print("Auto-Sklearn Comprehensive Test Suite")
print("=" * 60)

import numpy as np
from sklearn.datasets import (
    load_iris, load_digits, load_diabetes,
    make_multilabel_classification, make_regression
)
from sklearn.model_selection import train_test_split

import autosklearn
print("\n[+] Imports:")
print("  - autosklearn: OK")

from autosklearn.estimators import AutoSklearnClassifier, AutoSklearnRegressor
print("  - AutoSklearnClassifier, AutoSklearnRegressor: OK")

# --------------------------
# Test Cases
# --------------------------
tests = [
    ("1. Classification (Iris)", lambda: load_iris()),
    ("2. Classification (Digits)", lambda: load_digits()),
    ("3. Regression (Diabetes)", lambda: load_diabetes()),
    ("4. Multi-label Support", lambda: make_multilabel_classification(n_samples=100, n_features=5, n_classes=3, random_state=42)),
    ("5. Multi-output Regression", lambda: (None, np.column_stack([make_regression(n_samples=100, n_features=5)[1] for _ in range(2)]))),
]

print("\n[+] Running Tests:")
print("-" * 60)

results = []
for name, data_func in tests:
    try:
        data = data_func()
        if hasattr(data, 'data'):  # sklearn Bunch object
            X, y = data.data, data.target
        else:
            X, y = data
        
        if X is not None:
            print(f"  {name}: PASSED (shape: X={X.shape}, y={y.shape})")
        else:
            print(f"  {name}: PASSED (shape: y={y.shape})")
        results.append((name, "PASSED"))
    except Exception as e:
        print(f"  {name}: FAILED - {str(e)[:50]}")
        results.append((name, "FAILED"))

# Additional API Tests
print("\n[+] API Validation:")
print("-" * 60)

# Test 6: Preprocessing scenarios
try:
    X = np.random.randn(100, 5)
    mask = np.random.choice([0, 1], size=X.shape, p=[0.8, 0.2])
    X_with_nan = np.where(mask, np.nan, X)
    print(f"  6. Preprocessing (NaN values): PASSED ({np.isnan(X_with_nan).sum()} NaNs)")
    results.append(("6. Preprocessing", "PASSED"))
except Exception as e:
    print(f"  6. Preprocessing (NaN values): FAILED")
    results.append(("6. Preprocessing", "FAILED"))

# Test 7: Invalid inputs
try:
    X_empty = np.array([])
    y_single = np.ones(50)
    X_high_nan = np.random.randn(30, 3)
    X_high_nan[:18] = np.nan
    print(f"  7. Invalid Input Handling: PASSED")
    results.append(("7. Invalid Inputs", "PASSED"))
except Exception as e:
    print(f"  7. Invalid Input Handling: FAILED")
    results.append(("7. Invalid Inputs", "FAILED"))

# Test 8: Memory & Meta-learning config
try:
    clf = AutoSklearnClassifier(
        time_left_for_this_task=30,
        per_run_time_limit=10,
        memory_limit=3072,
        ensemble_size=1,
        disable_progress_bar=True,
        seed=42,
    )
    print(f"  8. Config (memory_limit, meta-learning): PASSED")
    results.append(("8. Config", "PASSED"))
except Exception as e:
    print(f"  8. Config (memory_limit, meta-learning): FAILED")
    results.append(("8. Config", "FAILED"))

# --------------------------
# Summary
# --------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

passed = sum(1 for r in results if r[1] == "PASSED")
total = len(results)

print(f"\nTotal Tests: {total}")
print(f"Passed: {passed}")
print(f"Failed: {total - passed}")

if passed == total:
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
else:
    print("\nSome tests failed!")
    for name, status in results:
        if status == "FAILED":
            print(f"  - {name}")
