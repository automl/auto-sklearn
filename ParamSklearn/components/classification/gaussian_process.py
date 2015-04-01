import numpy as np

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant

from sklearn.preprocessing import OneHotEncoder

from ParamSklearn.components.classification_base import ParamSklearnClassificationAlgorithm
from ParamSklearn.util import DENSE, PREDICTIONS



class GPyClassifier():#ParamSklearnClassificationAlgorithm):
    def __init__(self, random_state=None, n_inducing=5, ard=False):
        import GPy
        global GPy

        self.estimators = None
        self.n_inducing = int(n_inducing)

        if ard == "True":
            self.ard = True
        elif ard == "False":
            self.ard = False
        else:
            selfard = ard

        self.enc = None

    def fit(self, X, Y):
        # one hot encode targets for one against all classification
        self.enc = OneHotEncoder(sparse=False)
        targets = self.enc.fit_transform(Y[:,None])

        # create a list of GP models, one for each class
        self.estimators = []
        for i in range(self.enc.n_values_):
            # train model 
            kern = GPy.kern._src.rbf.RBF(X.shape[1], variance=1.0, lengthscale=1.0, ARD=self.ard)
            # dense
            # model = GPy.models.GPClassification(X, targets[:,i,None], kernel=kern)
            # sparse 
            model = GPy.models.SparseGPClassification(X, targets[:,i,None], kernel=kern, num_inducing=self.n_inducing)
            # fit kernel hyperparameters
            model.optimize('bfgs', max_iters=100)
            # add to list of estimators
            self.estimators.append(model)
        return self

    def predict(self, X):
        if self.estimators is None:
            raise NotImplementedError
        # get probabilities for each class
        probs = np.zeros([len(X), len(self.estimators)])
        for i, model in enumerate(self.estimators):
            probs[:,i] = model.predict(X)[0].flatten()        
        # return the most probable label 
        return self.enc.active_features_[np.argmax(probs, 1)]

    def predict_proba(self, X):
        if self.estimators is None:
            raise NotImplementedError()
        probs = np.zeros([len(X), len(self.estimators)])
        for i, model in enumerate(self.estimators):
            probs[:,i] = model.predict(X)[0].flatten()
        # normalize to get probabilities
        return probs / np.sum(probs,1)[:,None]

    @staticmethod
    def get_properties():
        return {'shortname': 'GPy',
                'name': 'Gaussian Process Classifier',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                # TODO find out if this is good because of sparcity...
                'prefers_data_normalized': False,
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                'handles_sparse': False,
                'input': (DENSE, ),
                'output': PREDICTIONS,
                # TODO find out what is best used here!
                'preferred_dtype': np.float32}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        ard = CategoricalHyperparameter("ard", ["True", "False"], default="False")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(ard)
        return cs

