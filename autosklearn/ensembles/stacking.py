# -*- encoding: utf-8 -*-
"""
Copyright 2015 University of Freiburg
Misgana Negassi <negassim@tf.uni-freiburg.de>
"""

import numpy as np
from cvxopt import matrix, solvers 
from scipy import optimize
import logging

from autosklearn.ensembles.abstract_ensemble import AbstractEnsemble

logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("stacking.py")
logger.setLevel(logging.DEBUG)

class Stacking(AbstractEnsemble):
    """
    """

    def __init__(self, ensemble_size, task_type, metric,
                 sorted_initialization=False, bagging=False, mode='fast'):
        pass


    @staticmethod
    def compute_y(l, N, true_labels):
        """
        compute_y checks  for each instance in the prediction_test if it and its true label,is in class l,  \
        and assign them to 1, and 0 Otherwise.
        """
        y = true_labels
        y[y != l] = 0
        y[y == l] = 1

        return y


    @staticmethod
    def compute_alpha(base_models_predictions, true_targets):

        """Computes weights of each model.

        Parameters
        ----------
        base_models_predictions : array of shape = [n_base_models, n_data_points]
        true_targets : array of shape [n_data_points]

        Returns
        -------
        alpha : array of shape = [n_base_models]

        """

        K = Z.shape[0]
        N = Z.shape[1]
        A = np.dot(Z, Z.transpose()) 
        b = np.dot(-Z, y)
        G = -np.eye(K)# negative identity matrix 
        H = np.zeros([K])
        # cvxopt expects double matrices
        sol_dict = solvers.qp(matrix(A.astype(np.double)), matrix(b.astype(np.double)), matrix(G), matrix(H)) 
        alpha = np.ravel(np.array(sol_dict['x']))

        return alpha

    def get_train_score(self):
        logger.warning('get_train_score not implemented.')
        return -1.

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
        Z = np.array(base_models_predictions) 
        y = true_targets
    
        L = Z.shape[0]
        N = Z.shape[1]
        K = Z.shape[2]
        self.Alpha = np.zeros([K, L])
        if not((y.shape[0] == N )):
            print("check if y is of shape Nx1")
        for l in range(0,  L):
            Zl = Z[l, :, :] # Do I need to remove singleton dimensions? no
            y_prob = Stacking.compute_y(l, N, y)
            alpha = Stacking.compute_alpha(Zl.transpose(), y_prob)
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
        array : [n_data_points, n_targets]
        """
        Z_test = np.array(base_models_predictions) 
        L = self.L
        K = self.K
        N_test = Z_test.shape[1]
        LR = np.zeros([N_test, K])           #LR : Is LxN_test matrix
        if not((Z_test.shape[0] == L) and (Z_test.shape[2] == K)):
            logger.error("check if predictions_test is of shape L x N_test x K")

        # for l in range(0, L):
        #     LR[l, :] = np.dot(self.Alpha[:, l].transpose() , Z_test[l].transpose()) 
        # return LR.T

        for k in range(0, K):
            LR[:, k] = np.dot(self.Alpha[k, :], Z_test[:, :, k]) 
        LR = LR / np.sum(LR, axis=1)[:, None] # normalize so we get probabilities
        return LR
        
       

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


