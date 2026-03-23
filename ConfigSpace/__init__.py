"""Mock ConfigSpace module for testing"""

class Configuration:
    def __init__(self, configuration_space, values=None):
        self.configuration_space = configuration_space
        self.values = values or {}
    
    def get_dictionary(self):
        return self.values
    
    def __getitem__(self, key):
        return self.values.get(key)
    
    def __contains__(self, key):
        return key in self.values

class ConfigurationSpace:
    def __init__(self, name=None, seed=None):
        self.name = name
        self.hyperparameters = {}
        self.conditions = []
        self.forbiddens = []
        self.random = seed
    
    def add_hyperparameter(self, hyperparameter):
        self.hyperparameters[hyperparameter.name] = hyperparameter
        return hyperparameter
    
    def add_hyperparameters(self, hyperparameters):
        for hp in hyperparameters:
            self.add_hyperparameter(hp)
    
    def add_condition(self, condition):
        self.conditions.append(condition)
    
    def add_conditions(self, conditions):
        for cond in conditions:
            self.add_condition(cond)
    
    def add_forbidden_clause(self, clause):
        self.forbiddens.append(clause)
    
    def get_hyperparameter(self, name):
        return self.hyperparameters.get(name)
    
    def get_hyperparameters(self):
        return list(self.hyperparameters.values())
    
    def get_hyperparameter_names(self):
        return list(self.hyperparameters.keys())
    
    def get_conditions(self):
        return self.conditions
    
    def get_forbiddens(self):
        return self.forbiddens
    
    def get_random_configuration(self):
        return Configuration(self)
    
    def check_configuration(self, configuration):
        return True
    
    def sample_configuration(self, size=1):
        if size == 1:
            return Configuration(self)
        return [Configuration(self) for _ in range(size)]

class Constant:
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.default_value = value

class UniformIntegerHyperparameter:
    def __init__(self, name, lower, upper, default_value=None, log=False):
        self.name = name
        self.lower = lower
        self.upper = upper
        self.log = log
        self.default_value = default_value if default_value is not None else lower

class UniformFloatHyperparameter:
    def __init__(self, name, lower, upper, default_value=None, log=False):
        self.name = name
        self.lower = lower
        self.upper = upper
        self.log = log
        self.default_value = default_value if default_value is not None else lower

class CategoricalHyperparameter:
    def __init__(self, name, choices, default_value=None, weights=None):
        self.name = name
        self.choices = list(choices)
        self.weights = weights
        self.default_value = default_value if default_value is not None else self.choices[0]

# Import conditions
from .conditions import (
    Condition,
    AbstractCondition,
    AbstractValueCondition,
    EqualsCondition,
    InCondition,
    AndConjunction,
    OrConjunction,
)

# Additional condition types
class NotEqualsCondition(AbstractValueCondition):
    def __init__(self, hyperparameter, parent_hyperparameter, value):
        super().__init__(hyperparameter, parent_hyperparameter, value)

class GreaterThanCondition(AbstractValueCondition):
    def __init__(self, hyperparameter, parent_hyperparameter, value):
        super().__init__(hyperparameter, parent_hyperparameter, value)

class LessThanCondition(AbstractValueCondition):
    def __init__(self, hyperparameter, parent_hyperparameter, value):
        super().__init__(hyperparameter, parent_hyperparameter, value)

# Import forbidden
from .forbidden import (
    AbstractForbiddenClause,
    AbstractForbiddenValueClause,
    ForbiddenEqualsClause,
    ForbiddenInClause,
    ForbiddenRelation,
    AbstractForbiddenConjunction,
    ForbiddenAndConjunction,
)

# Import hyperparameters
from .hyperparameters import (
    UniformInteger,
    UniformFloat,
    Categorical,
    UnParametrizedHyperparameter,
    NumericalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
)

__all__ = [
    'Configuration',
    'ConfigurationSpace',
    'Constant',
    'UniformIntegerHyperparameter',
    'UniformFloatHyperparameter',
    'CategoricalHyperparameter',
    'Condition',
    'AbstractCondition',
    'AbstractValueCondition',
    'EqualsCondition',
    'NotEqualsCondition',
    'GreaterThanCondition',
    'LessThanCondition',
    'InCondition',
    'AndConjunction',
    'OrConjunction',
    'AbstractForbiddenClause',
    'AbstractForbiddenValueClause',
    'ForbiddenEqualsClause',
    'ForbiddenInClause',
    'ForbiddenRelation',
    'AbstractForbiddenConjunction',
    'ForbiddenAndConjunction',
    'UniformInteger',
    'UniformFloat',
    'Categorical',
    'UnParametrizedHyperparameter',
    'NumericalHyperparameter',
    'FloatHyperparameter',
    'IntegerHyperparameter',
]
