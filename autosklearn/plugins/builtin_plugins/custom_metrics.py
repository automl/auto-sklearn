"""Custom metric plugins for auto-sklearn."""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from autosklearn.plugins import MetricPlugin, register_plugin


@register_plugin
class F1ScoreMacro(MetricPlugin):
    """F1 Score with macro averaging."""
    
    @classmethod
    def get_name(cls) -> str:
        return "f1_macro"
    
    def __call__(self, y_true, y_pred, **kwargs) -> float:
        return f1_score(y_true, y_pred, average="macro")


@register_plugin
class F1ScoreMicro(MetricPlugin):
    """F1 Score with micro averaging."""
    
    @classmethod
    def get_name(cls) -> str:
        return "f1_micro"
    
    def __call__(self, y_true, y_pred, **kwargs) -> float:
        return f1_score(y_true, y_pred, average="micro")


@register_plugin
class ROCAUCOneVsRest(MetricPlugin):
    """ROC AUC Score with one-vs-rest strategy for multi-class classification."""
    
    @classmethod
    def get_name(cls) -> str:
        return "roc_auc_ovr"
    
    def __call__(self, y_true, y_pred, **kwargs) -> float:
        from sklearn.preprocessing import label_binarize
        
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            return roc_auc_score(y_true, y_pred)
        
        # For multi-class, we need probabilities
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        return roc_auc_score(y_true_bin, y_pred, multi_class="ovr")


@register_plugin
class MatthewsCorrelationCoefficient(MetricPlugin):
    """Matthews Correlation Coefficient for binary classification."""
    
    @classmethod
    def get_name(cls) -> str:
        return "matthews_corrcoef"
    
    def __call__(self, y_true, y_pred, **kwargs) -> float:
        from sklearn.metrics import matthews_corrcoef
        return matthews_corrcoef(y_true, y_pred)


@register_plugin
class MeanSquaredError(MetricPlugin):
    """Mean Squared Error for regression tasks."""
    
    @classmethod
    def get_name(cls) -> str:
        return "mse"
    
    def __call__(self, y_true, y_pred, **kwargs) -> float:
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(y_true, y_pred)
    
    @classmethod
    def is_better(cls, value1: float, value2: float) -> bool:
        """For MSE, lower is better."""
        return value1 < value2


@register_plugin
class MeanAbsoluteError(MetricPlugin):
    """Mean Absolute Error for regression tasks."""
    
    @classmethod
    def get_name(cls) -> str:
        return "mae"
    
    def __call__(self, y_true, y_pred, **kwargs) -> float:
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error(y_true, y_pred)
    
    @classmethod
    def is_better(cls, value1: float, value2: float) -> bool:
        """For MAE, lower is better."""
        return value1 < value2
