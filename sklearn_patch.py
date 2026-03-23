"""Patch sklearn utils for compatibility"""
import sys
import sklearn.utils

# Add missing _approximate_mode if it doesn't exist
if not hasattr(sklearn.utils, '_approximate_mode'):
    def _approximate_mode(class_counts, n_draws, rng):
        """Approximate mode sampling for sklearn compatibility"""
        from scipy.stats import poisson
        pvals = class_counts / class_counts.sum()
        return rng.choice(len(pvals), size=n_draws, p=pvals)
    
    sklearn.utils._approximate_mode = _approximate_mode
    print("Patched sklearn.utils._approximate_mode")

# Also patch for any other missing utilities
if not hasattr(sklearn.utils, 'safe_indexing'):
    from sklearn.utils import _safe_indexing
    sklearn.utils.safe_indexing = _safe_indexing
    print("Patched sklearn.utils.safe_indexing")
