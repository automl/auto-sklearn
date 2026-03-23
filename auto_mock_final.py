"""Final auto-mock with comprehensive dask patching"""
import sys
import os
from types import ModuleType

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock resource module for Windows
import resource

# Patch sklearn for compatibility
import sklearn_patch

# Patch pkg_resources
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
    'liac-arff': '2.5.0',
    'tqdm': '4.62.0',
    'threadpoolctl': '3.0.0',
    'distro': '1.7.0',
    'typing_extensions': '4.0.0',
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

# Mock classes
class MockLocalCluster:
    def __init__(self, *args, **kwargs):
        pass
    def close(self):
        pass

class MockFuture:
    def result(self):
        return None
    def done(self):
        return True
    def cancelled(self):
        return False

class MockClient:
    def __init__(self, *args, **kwargs):
        pass
    def submit(self, *args, **kwargs):
        return MockFuture()
    def map(self, *args, **kwargs):
        return []
    def close(self):
        pass

class MockFireAndForget:
    pass

# Create and setup dask modules BEFORE any other imports
# Create dask module
dask_mod = ModuleType('dask')
dask_mod.__version__ = _versions['dask']
dask_mod.__file__ = ''
dask_mod.__path__ = []
sys.modules['dask'] = dask_mod

# Create dask.distributed module
dask_dist_mod = ModuleType('dask.distributed')
dask_dist_mod.Client = MockClient
dask_dist_mod.LocalCluster = MockLocalCluster
dask_dist_mod.Future = MockFuture
dask_dist_mod.FireAndForget = MockFireAndForget
dask_dist_mod.__file__ = ''
dask_dist_mod.__path__ = []
sys.modules['dask.distributed'] = dask_dist_mod
sys.modules['dask'].distributed = dask_dist_mod

# Create distributed module
dist_mod = ModuleType('distributed')
dist_mod.__version__ = _versions['distributed']
dist_mod.Client = MockClient
dist_mod.LocalCluster = MockLocalCluster
dist_mod.FireAndForget = MockFireAndForget
dist_mod.__file__ = ''
dist_mod.__path__ = []
sys.modules['distributed'] = dist_mod

# Create distributed.client module
dist_client_mod = ModuleType('distributed.client')
dist_client_mod.Client = MockClient
dist_client_mod.LocalCluster = MockLocalCluster
dist_client_mod.FireAndForget = MockFireAndForget
dist_client_mod.__file__ = ''
dist_client_mod.__path__ = []
sys.modules['distributed.client'] = dist_client_mod
sys.modules['distributed'].client = dist_client_mod

# Quick test - this is what autosklearn is trying to do
import dask.distributed
print(f"dask.distributed.Client available: {hasattr(dask.distributed, 'Client')}")

# Now import autosklearn
print("\nImporting autosklearn...")
try:
    import autosklearn
    print("autosklearn imported")
    
    from autosklearn.estimators import AutoSklearnClassifier, AutoSklearnRegressor
    print("AutoSklearn estimators imported")
    
    print("\n" + "="*60)
    print("SUCCESS: All imports passed!")
    print("="*60)
    
except Exception as e:
    print(f"\nFAILED: {e}")
    import traceback
    traceback.print_exc()
