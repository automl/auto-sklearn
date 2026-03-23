"""Mock ConfigSpace.conditions module"""
class Condition:
    pass

class AbstractCondition(Condition):
    def __init__(self, hyperparameter, parent_hyperparameter):
        self.hyperparameter = hyperparameter
        self.parent = parent_hyperparameter

class AbstractValueCondition(AbstractCondition):
    def __init__(self, hyperparameter, parent_hyperparameter, value):
        super().__init__(hyperparameter, parent_hyperparameter)
        self.value = value

class EqualsCondition(AbstractValueCondition):
    pass

class InCondition(AbstractCondition):
    def __init__(self, hyperparameter, parent_hyperparameter, values):
        super().__init__(hyperparameter, parent_hyperparameter)
        self.values = values

class AndConjunction(Condition):
    def __init__(self, *conditions):
        self.conditions = conditions

class OrConjunction(Condition):
    def __init__(self, *conditions):
        self.conditions = conditions
