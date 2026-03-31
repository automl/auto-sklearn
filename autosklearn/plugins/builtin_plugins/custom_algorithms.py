"""Custom algorithm plugins for auto-sklearn."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression as SKLearnLogisticRegression
from sklearn.linear_model import Ridge as SKLearnRidge
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder

from autosklearn.plugins import AlgorithmPlugin, register_plugin


class CustomLogisticRegression(BaseEstimator, ClassifierMixin, AlgorithmPlugin):
    """Custom Logistic Regression classifier with additional regularization options."""
    
    @classmethod
    def get_name(cls) -> str:
        return "custom_logistic_regression"
    
    def __init__(
        self,
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=100,
        class_weight=None,
        random_state=None,
    ):
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self._model = None
        self._le = LabelEncoder()
    
    def fit(self, X, y, **kwargs):
        """Fit the model to the training data."""
        X, y = check_X_y(X, y)
        
        # Encode labels
        y_encoded = self._le.fit_transform(y)
        self.classes_ = self._le.classes_
        
        # Create and fit the model
        self._model = SKLearnLogisticRegression(
            penalty=self.penalty,
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self._model.fit(X, y_encoded)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X, **kwargs):
        """Make predictions."""
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        
        y_pred_encoded = self._model.predict(X)
        return self._le.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X, **kwargs):
        """Predict class probabilities."""
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        return self._model.predict_proba(X)


class CustomRidgeRegression(BaseEstimator, RegressorMixin, AlgorithmPlugin):
    """Custom Ridge Regression with additional features."""
    
    @classmethod
    def get_name(cls) -> str:
        return "custom_ridge_regression"
    
    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True,
        solver="auto",
        random_state=None,
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.random_state = random_state
        self._model = None
    
    def fit(self, X, y, **kwargs):
        """Fit the model to the training data."""
        X, y = check_X_y(X, y, y_numeric=True)
        
        self._model = SKLearnRidge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            solver=self.solver,
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X, **kwargs):
        """Make predictions."""
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        return self._model.predict(X)


class SimpleEnsemble(BaseEstimator, ClassifierMixin, AlgorithmPlugin):
    """A simple ensemble that averages predictions from multiple models."""
    
    @classmethod
    def get_name(cls) -> str:
        return "simple_ensemble"
    
    def __init__(
        self,
        n_estimators=5,
        base_estimator=None,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator or SKLearnLogisticRegression()
        self.random_state = random_state
        self._estimators = []
    
    def fit(self, X, y, **kwargs):
        """Fit the ensemble to the training data."""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        
        rng = np.random.RandomState(self.random_state)
        
        for i in range(self.n_estimators):
            # Bootstrap sample
            indices = rng.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            estimator = SKLearnLogisticRegression(random_state=rng.randint(10000))
            estimator.fit(X_bootstrap, y_bootstrap)
            self._estimators.append(estimator)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X, **kwargs):
        """Make predictions by majority voting."""
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        
        predictions = np.array([estimator.predict(X) for estimator in self._estimators])
        
        # Majority voting
        y_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions
        )
        
        return y_pred
    
    def predict_proba(self, X, **kwargs):
        """Predict class probabilities by averaging."""
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        
        probabilities = np.array([estimator.predict_proba(X) for estimator in self._estimators])
        return np.mean(probabilities, axis=0)


# Register the plugins
register_plugin(CustomLogisticRegression)
register_plugin(CustomRidgeRegression)
register_plugin(SimpleEnsemble)
