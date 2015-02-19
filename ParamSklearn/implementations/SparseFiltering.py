"""
This quickly adapted version of sparse filtering requires scipy and numpy
"""
import numpy as np
from scipy.optimize import minimize

def l2row(X):
    """
    L2 normalize X by rows. We also use this to normalize by column with l2row(X.T)
    """
    N = np.sqrt((X**2).sum(axis=1)+1e-8)
    Y = (X.T/N).T
    return Y,N


def l2rowg(X,Y,N,D):
    """
    Compute L2 normalized gradient.
    """
    return (D.T/N - Y.T * (D*X).sum(axis=1) / N**2).T


class SparseFiltering(object):
    def __init__(self, N, maxiter=100, random_state=None):
        self.N = N
        self.W = None
        self.maxiter = maxiter
        if random_state is None:
            self.rng = np.random
        elif isinstance(random_state, int):
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = random_state

    def step(self, X, W):
        # returns current objective and gradient
        W = W.reshape((X.shape[1], self.N))
        features = X.dot(W) #W.dot(X)
        features_norm = np.sqrt(features**2 + 1e-8)
        features_column, column_norm = l2row(features_norm.T)
        features_row, row_norm = l2row(features_norm)
        # compute objective function (l1 norm of features)
        obj = features_row.sum()
        # backprop through the whole process
        deltaW = l2rowg(features_norm, features_row, row_norm, np.ones(features_row.shape))
        deltaW = l2rowg(features_norm.T, features_column, column_norm, deltaW.T).T
        deltaW = X.T.dot(deltaW*(features/features_norm))
        return obj, deltaW.flatten()

        
    def fit(self, X, y=None):
        """ fit sparse filtering to data
           this completely ignores y
        """
        # init random weights
        W = self.rng.randn(self.N,X.shape[1])
        # build a closure for the objective 
        obj_fun = lambda w: self.step(X, w)
        # evaluate once for testing
        obj, grad = obj_fun(W)
        # and run optimization
        opt = {'maxiter': self.maxiter}
        res = minimize(obj_fun, W, method='L-BFGS-B', jac = True, options = opt)
        self.W = res.x.reshape(X.shape[1], self.N)

    def transform(self, X):
        # compute responses
        features = X.dot(self.W)
        # sparsify
        features_norm = np.sqrt(features**2 + 1e-8)
        features_column = l2row(features_norm.T)[0]
        features_row = l2row(features_column)[0].T
        return features_row
