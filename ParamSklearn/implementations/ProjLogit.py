import numpy as np
import numpy.random as npr

# from http://arxiv.org/pdf/1309.1541v1.pdf
def proj_simplex(Y):
    N,D = np.shape(Y)
    # sort in descending order
    X = -np.sort(-Y)
    Xsum = np.cumsum(X, axis = 1) - 1
    Xsum = Xsum * (1./np.arange(1,D+1))
    biggest = np.sum(X > Xsum, axis = 1)
    # TODO last step could be made faster
    #      via ravel / linear indexing
    subtract = np.zeros((N, 1))
    for i in range(N):
        subtract[i] = Xsum[i, biggest[i]-1]
    return np.maximum(Y - subtract, 0)


class ProjLogit(object):

    def __init__(self, max_epochs = 10, verbose = False):
        self.w = None
        self.ws = None
        self.max_epochs = max_epochs
        self.verbose = verbose

    def fit(self, X, Y):
        # get one hot encoding and add a bias
        n = X.shape[0]
        trainx = np.hstack([np.ones((n, 1)), X])
        k = np.max(Y) + 1
        if self.verbose:
            print("Using {} samples of {} classes".format(n,k))
        yt = np.zeros((n, k))
        for i in range(n):
            yt[i, Y[i]] = 1
        # initialize with linear regression
        precond = np.eye(trainx.shape[1]) * np.sqrt(n)
        C = np.linalg.cholesky(0.5 * np.dot(trainx.T,trainx) + precond)
        wp = np.linalg.solve(C, np.dot(trainx.T, yt))
        w = np.linalg.solve(C.T, wp)
        pred_train = np.dot(trainx, w)
        for i in range(self.max_epochs):
            # expand prediction
            res = np.hstack([pred_train, np.power(pred_train, 2) / 2., np.power(pred_train, 3) / 6., np.power(pred_train, 4) / 24.])
            # solve with linear regression
            precond = np.eye(res.shape[1]) * np.sqrt(n)
            Cp = np.linalg.cholesky(np.dot(res.T,res) + precond)
            ws = np.linalg.solve(Cp.T, np.linalg.solve(Cp, np.dot(res.T, yt)))
            # project to probability simplex
            p_res = proj_simplex(np.dot(res, ws))
            # and solve again with updated residual
            wp = np.linalg.solve(C, np.dot(trainx.T, (yt - p_res)))
            w = np.linalg.solve(C.T, wp)
            pred_train = p_res + np.dot(trainx, w)
            obj = np.linalg.norm(yt - pred_train)

            # compute train error
            errort = np.sum(np.argmax(pred_train, axis = 1) != Y)
            # print training error
            if self.verbose:
                print("Epoch {} obj: {} train error: {}".format(i,obj,1.*errort/n))
        self.ws = ws
        self.w = w
        return self

    def predict(self, X):
        testx = np.hstack([np.ones((X.shape[0], 1)), X])
        pred = np.dot(testx, self.w)
        res = np.argmax(pred, axis = 1)
        return res

    def predict_proba(self, X):
        if self.w == None:
            return np.zeros(X.shape[0])
        testx = np.hstack([np.ones((X.shape[0], 1)), X])
        pred = np.dot(testx, self.w)
        #print(pred)
        p_res = proj_simplex(pred)
        return p_res
    
    def predict_log_proba(self, X):
        if self.w == None:
            return np.zeros(X.shape[0])
        res = np.log(self.predict_proba(X))
        return res
