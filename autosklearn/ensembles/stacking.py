# -*- encoding: utf-8 -*-

import numpy as np
from cvxopt import matrix, solvers 
from scipy import optimize

from autosklearn.ensembles.abstract_ensemble import AbstractEnsemble

class Stacking(AbstractEnsemble):

    def __init__(self, ensemble_size, task_type, metric,
                 sorted_initialization=False, bagging=False, mode='fast'):
        pass


    @staticmethod
    def compute_y(l, N, true_labels):
        """
        compute_y checks  for each instance in the prediction_test if it and its true label,is in class l,  \
        and assign them to 1, and 0 Otherwise.

         TODO: code can be more effiecient.
        """
        y = np.zeros([N]) 
        for i in range(0, N):
            if(true_labels[i] == l):
                y[i] = 1
            else:
                y[i] = 0 
        return y


    @staticmethod
    def compute_alpha(Z, y):
        """ 
        Z is matrix of K- predictors on size n dataset 
        y : Is a vector of true class labels of size [N,1]
        alpha: is a vector of size K
        """

        K = Z.shape[0]
        N = Z.shape[1]
        A = np.zeros([K , K])
        b = np.zeros([K])

        A = np.dot(Z,  Z.transpose()) 
        b = np.dot(-Z , y)
        G = -np.eye(K)# negative identity matrix 
        H = np.zeros([K])
        sol_dict = solvers.qp(matrix(A), matrix(b),matrix(G) , matrix(H)) 
        alpha =np.ravel(np.array(sol_dict['x']))
        #alpha, _ = optimize.nnls(A, b) # ignore rnorm output
        return alpha

    def fit(self, base_models_predictions, true_targets, model_identifiers):
        """Fit an ensemble given predictions of base models and targets.

        Parameters
        ----------
        base_models_predictions : array of shape = [n_base_models, n_data_points, n_targets]
            n_targets is the number of classes in case of classification,
            n_targets is 0 or 1 in case of regression

        true_targets : array of shape [n_targets]

        model_identifiers : identifier for each base model.
            Can be used for practical text output of the ensemble.

        Returns
        -------
        self

        """
        self.model_identifiers = model_identifiers
        Z = base_models_predictions
        y = true_targets
        L = Z.shape[0]
        K = Z.shape[1]
        N = Z.shape[2]
        self.Alpha = np.zeros([K, L])
        if not((y.shape[0] == N )):
            print("check if y is of shape Nx1")
        for l in range(0,  L):
            Zl = Z[l, :, :] # Do I need to remove singleton dimensions? no
            y_prob = compute_y(l, N, y)
            alpha = compute_alpha(Zl, y_prob)
            self.Alpha[:, l] = alpha

        self.L = L # save these for consitency checks when making predictions
        self.K = K
        return self # return self for compliance

    def predict(self, base_models_predictions):
        """Create ensemble predictions from the base model predictions.

        Parameters
        ----------
        base_models_predictions : array of shape = [n_base_models, n_data_points, n_targets]
            Same as in the fit method.

        Returns
        -------
        array : [n_data_points]
        """
        Z_test = base_models_predictions
        probas = self.predict_proba(Z_test)
        return np.argmax(probas, axis=1) # for each classes, return the max value LR on a test instance

    def predict_proba(self, base_models_predictions):
        """Create ensemble predictions from the base model predictions.

        Parameters
        ----------
        base_models_predictions : array of shape = [n_base_models, n_data_points, n_targets]
            Same as in the fit method.

        Returns
        -------
        array : [n_data_points]
        """
        Z = base_models_predictions
        L = self.L
        K = self.K
        N_test = Z_test.shape[2]
        LR = np.zeros([L, N_test])           #LR : Is LxN_test matrix
        if not((Z_test.shape[0] == L) and (Z_test.shape[1] == K)):
            print("check if predictions_test is of shape LxKxN_test")

        for l in range(0, L):
                LR[l, :] = np.dot(self.Alpha[:, l].transpose() , Z_test[l]) 
        return LR.T

       

    def pprint_ensemble_string(self, models):
        """Return a nicely-readable representation of the ensmble.

        Parameters
        ----------
        models : dict {identifier : model object}
            The identifiers are the same as the one presented to the fit()
            method. Models can be used for nice printing.

        Returns
        -------
        str
        """
        output = []
        sio = six.StringIO()
        for i in range(0, self.K):
            identifier = self.model_identifiers[i]
            model = models[identifier]
            output.append((self.Alpha[:,i], model))

        #output.sort(reverse=True)

        sio.write("[")
        for alpha, model in output:
            sio.write("(%s, %s),\n" % (np.array_str(alpha), model))
        sio.write("]")

        return sio.getvalue()
    
    def get_model_identifiers(self):
        """Return identifiers of models in the ensemble.

        This includes models which have a weight of zero!

        Returns
        -------
        list
        """
        return self.model_identifiers


