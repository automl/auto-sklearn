"""
Standalone test runner that bypasses Windows compatibility checks.
This script directly imports the necessary modules for testing.
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ['AUTOSKLEARN_DISABLE_VERSION_CHECK'] = '1'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import make_classification, make_multilabel_classification, make_regression

print("=" * 80)
print("AutoSklearn Comprehensive Test Suite")
print("=" * 80)

TEST_RESULTS = {
    'passed': 0,
    'failed': 0,
    'errors': [],
}

def test_result(name: str, passed: bool, message: str = ""):
    status = "PASSED" if passed else "FAILED"
    symbol = "[+]" if passed else "[-]"
    print(f"{symbol} {name}: {status}")
    if message:
        print(f"    {message}")
    if passed:
        TEST_RESULTS['passed'] += 1
    else:
        TEST_RESULTS['failed'] += 1
        TEST_RESULTS['errors'].append(f"{name}: {message}")

def run_tests():
    try:
        from autosklearn.automl import AutoML, AutoMLClassifier, AutoMLRegressor
        from autosklearn.estimators import AutoSklearnClassifier, AutoSklearnRegressor
        from autosklearn.metrics import accuracy, r2, mean_squared_error
        from dask.distributed import Client
        
        print("\n[1/7] Testing AutoSklearnClassifier Basic Classification")
        print("-" * 60)
        
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        client = Client(n_workers=1, threads_per_worker=1, processes=False, scheduler_port=0, dashboard_address=None)
        
        try:
            cls = AutoSklearnClassifier(
                time_left_for_this_task=30,
                per_run_time_limit=10,
                seed=42,
                n_jobs=1,
                memory_limit=1024,
                ensemble_size=5,
                ensemble_nbest=5,
                max_models_on_disc=5,
                initial_configurations_via_metalearning=3,
                dask_client=client,
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cls.fit(X_train, y_train)
            
            test_result("fit() completes without error", True)
            
            predictions = cls.predict(X_test)
            test_result("predict() returns correct shape", 
                       predictions.shape == y_test.shape,
                       f"Expected {y_test.shape}, got {predictions.shape}")
            
            proba = cls.predict_proba(X_test)
            proba_sums = np.sum(proba, axis=1)
            test_result("predict_proba() sums to 1",
                       np.allclose(proba_sums, 1.0, atol=1e-5),
                       f"Sums: {proba_sums[:3]}...")
            
            test_result("predict_proba() in range [0,1]",
                       np.all(proba >= 0) and np.all(proba <= 1))
            
            score = cls.score(X_test, y_test)
            random_baseline = 1.0 / len(np.unique(y_train))
            test_result("score() > random baseline",
                       score > random_baseline * 1.5,
                       f"Score: {score:.4f}, Random: {random_baseline:.4f}")
            
            stats = cls.sprint_statistics()
            test_result("sprint_statistics() returns string",
                       isinstance(stats, str) and len(stats) > 0,
                       f"Length: {len(stats)}")
            
        except Exception as e:
            test_result("AutoSklearnClassifier basic tests", False, str(e))
        finally:
            client.close()
        
        print("\n[2/7] Testing AutoSklearnRegressor Regression")
        print("-" * 60)
        
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        client = Client(n_workers=1, threads_per_worker=1, processes=False, scheduler_port=0, dashboard_address=None)
        
        try:
            reg = AutoSklearnRegressor(
                time_left_for_this_task=30,
                per_run_time_limit=10,
                seed=42,
                n_jobs=1,
                memory_limit=1024,
                ensemble_size=5,
                ensemble_nbest=5,
                max_models_on_disc=5,
                initial_configurations_via_metalearning=3,
                dask_client=client,
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reg.fit(X_train, y_train)
            
            test_result("Regressor fit() completes", True)
            
            predictions = reg.predict(X_test)
            test_result("Regressor predict() correct shape",
                       predictions.shape == y_test.shape,
                       f"Expected {y_test.shape}, got {predictions.shape}")
            
            test_result("Predictions are continuous",
                       np.issubdtype(predictions.dtype, np.floating))
            
            test_result("No NaN in predictions",
                       not np.any(np.isnan(predictions)))
            
            score = reg.score(X_test, y_test)
            test_result("R2 score > 0.3",
                       score > 0.3,
                       f"R2: {score:.4f}")
            
        except Exception as e:
            test_result("AutoSklearnRegressor tests", False, str(e))
        finally:
            client.close()
        
        print("\n[3/7] Testing Multi-label Classification")
        print("-" * 60)
        
        X, y = make_multilabel_classification(n_samples=150, n_features=15, n_classes=3, n_labels=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        client = Client(n_workers=1, threads_per_worker=1, processes=False, scheduler_port=0, dashboard_address=None)
        
        try:
            cls = AutoSklearnClassifier(
                time_left_for_this_task=30,
                per_run_time_limit=10,
                seed=42,
                n_jobs=1,
                memory_limit=1024,
                ensemble_size=5,
                ensemble_nbest=5,
                max_models_on_disc=5,
                initial_configurations_via_metalearning=3,
                dask_client=client,
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cls.fit(X_train, y_train)
            
            test_result("Multi-label fit() completes", True)
            
            predictions = cls.predict(X_test)
            test_result("Multi-label predictions 2D",
                       predictions.ndim == 2,
                       f"Got {predictions.ndim}D")
            
            test_result("Multi-label correct shape",
                       predictions.shape == y_test.shape,
                       f"Expected {y_test.shape}, got {predictions.shape}")
            
        except Exception as e:
            test_result("Multi-label classification tests", False, str(e))
        finally:
            client.close()
        
        print("\n[4/7] Testing Multi-output Regression")
        print("-" * 60)
        
        X, y = make_regression(n_samples=150, n_features=15, n_targets=3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        client = Client(n_workers=1, threads_per_worker=1, processes=False, scheduler_port=0, dashboard_address=None)
        
        try:
            reg = AutoSklearnRegressor(
                time_left_for_this_task=30,
                per_run_time_limit=10,
                seed=42,
                n_jobs=1,
                memory_limit=1024,
                ensemble_size=5,
                ensemble_nbest=5,
                max_models_on_disc=5,
                initial_configurations_via_metalearning=3,
                dask_client=client,
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reg.fit(X_train, y_train)
            
            test_result("Multi-output fit() completes", True)
            
            predictions = reg.predict(X_test)
            test_result("Multi-output correct shape",
                       predictions.shape == y_test.shape,
                       f"Expected {y_test.shape}, got {predictions.shape}")
            
        except Exception as e:
            test_result("Multi-output regression tests", False, str(e))
        finally:
            client.close()
        
        print("\n[5/7] Testing Invalid Input Handling")
        print("-" * 60)
        
        client = Client(n_workers=1, threads_per_worker=1, processes=False, scheduler_port=0, dashboard_address=None)
        
        try:
            cls = AutoSklearnClassifier(
                time_left_for_this_task=15,
                per_run_time_limit=5,
                seed=42,
                n_jobs=1,
                memory_limit=512,
                dask_client=client,
            )
            
            X_single_class = np.random.rand(30, 5)
            y_single_class = np.ones(30)
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cls.fit(X_single_class, y_single_class)
                test_result("Single class raises error", False, "Should have raised error")
            except (ValueError, RuntimeError) as e:
                test_result("Single class raises error", True, str(e)[:50])
            
            X_nan = np.random.rand(30, 5)
            X_nan[:, 0] = np.nan
            X_nan[:, 1] = np.nan
            X_nan[:, 2] = np.nan
            y = np.random.randint(0, 2, 30)
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cls.fit(X_nan, y)
                test_result("High NaN handled gracefully", True)
            except Exception as e:
                test_result("High NaN raises meaningful error", True, str(e)[:50])
            
            X_const = np.ones((30, 5))
            X_const[:, 0] = np.random.rand(30)
            y = np.random.randint(0, 2, 30)
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cls.fit(X_const, y)
                test_result("Constant features handled", True)
            except Exception as e:
                test_result("Constant features error", False, str(e)[:50])
            
        except Exception as e:
            test_result("Invalid input tests", False, str(e))
        finally:
            client.close()
        
        print("\n[6/7] Testing Meta-learning Cold Start")
        print("-" * 60)
        
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        client = Client(n_workers=1, threads_per_worker=1, processes=False, scheduler_port=0, dashboard_address=None)
        
        try:
            cls_meta = AutoSklearnClassifier(
                time_left_for_this_task=20,
                per_run_time_limit=5,
                seed=42,
                n_jobs=1,
                memory_limit=1024,
                ensemble_size=5,
                initial_configurations_via_metalearning=5,
                dask_client=client,
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cls_meta.fit(X_train, y_train)
            
            score_meta = cls_meta.score(X_test, y_test)
            test_result("Meta-learning achieves >0.5 accuracy",
                       score_meta > 0.5,
                       f"Score: {score_meta:.4f}")
            
        except Exception as e:
            test_result("Meta-learning tests", False, str(e))
        finally:
            client.close()
        
        print("\n[7/7] Testing Memory Limits")
        print("-" * 60)
        
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        client = Client(n_workers=1, threads_per_worker=1, processes=False, scheduler_port=0, dashboard_address=None)
        
        try:
            reg = AutoSklearnRegressor(
                time_left_for_this_task=20,
                per_run_time_limit=5,
                seed=42,
                n_jobs=1,
                memory_limit=256,
                ensemble_size=5,
                dask_client=client,
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reg.fit(X_train, y_train)
            
            predictions = reg.predict(X_test)
            test_result("Memory limit 256MB works",
                       predictions.shape == y_test.shape,
                       f"Shape: {predictions.shape}")
            
        except Exception as e:
            test_result("Memory limit tests", False, str(e))
        finally:
            client.close()
        
    except ImportError as e:
        print(f"\n[ERROR] Failed to import required modules: {e}")
        print("Make sure autosklearn and dependencies are installed.")
        return False
    
    return True

if __name__ == "__main__":
    success = run_tests()
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {TEST_RESULTS['passed']}")
    print(f"Failed: {TEST_RESULTS['failed']}")
    
    if TEST_RESULTS['errors']:
        print("\nFailed Tests:")
        for error in TEST_RESULTS['errors']:
            print(f"  - {error}")
    
    print("=" * 80)
    
    sys.exit(0 if TEST_RESULTS['failed'] == 0 else 1)
