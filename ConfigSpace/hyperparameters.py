"""Mock ConfigSpace.hyperparameters module"""
from . import (
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    Constant,
)

__all__ = [
    'CategoricalHyperparameter',
    'UniformIntegerHyperparameter',
    'UniformFloatHyperparameter',
    'Constant',
    'UniformInteger',
    'UniformFloat',
    'Categorical',
    'UnParametrizedHyperparameter',
    'NumericalHyperparameter',
    'FloatHyperparameter',
    'IntegerHyperparameter',
]

# Aliases
UniformInteger = UniformIntegerHyperparameter
UniformFloat = UniformFloatHyperparameter
Categorical = CategoricalHyperparameter
UnParametrizedHyperparameter = Constant
NumericalHyperparameter = UniformFloatHyperparameter
FloatHyperparameter = UniformFloatHyperparameter
IntegerHyperparameter = UniformIntegerHyperparameter
