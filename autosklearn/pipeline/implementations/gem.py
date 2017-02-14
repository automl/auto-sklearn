import numpy as np
from scipy.sparse.linalg import eigs


class GEM(object):


    def __init__(self, N, precond):
        self.N = N
        self.precond = precond
        self.W = None
        self.verbose = False


    def fit(self, X, Y):
        self.N = min(self.N, X.shape[1]-2)
        y_max = int(np.max(Y) + 1)
        self.W = np.zeros((X.shape[1], self.N*y_max*(y_max-1)), dtype=X.dtype)
        off = 0
        for i in range(y_max):
            Xi = X[Y == i]
            covi = np.dot(Xi.T, Xi)
            covi /= np.float32(Xi.shape[0])
            for j in range(y_max):
                if j == i:
                    continue
                if self.verbose:
                    print("Finding eigenvectors for pair ({}/{})".format(i,j))
                Xj = X[Y == j]
                covj = np.dot(Xj.T, Xj) / np.float32(Xj.shape[0])
                E = np.linalg.pinv(np.linalg.cholesky(covj + np.eye(covj.shape[0]) * self.precond).T)
                C = np.dot(np.dot(E.T, covi), E)
                C2 = 0.5 * (C + C.T)
                S,U = eigs(C2, self.N)
                gev = np.dot(E, U[:, :self.N])
                self.W[:, off:off+self.N] = np.real(gev)
                off += self.N
        if self.verbose:
            print("DONE")
        return self

    
    def transform(self, X, Y=None):
        features = np.maximum(np.dot(X, self.W), 0)
        return features
                
            
