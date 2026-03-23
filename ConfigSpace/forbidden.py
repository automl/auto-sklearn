"""Mock ConfigSpace.forbidden module"""
class AbstractForbiddenClause:
    def __init__(self, hyperparameter):
        self.hyperparameter = hyperparameter

class AbstractForbiddenValueClause(AbstractForbiddenClause):
    def __init__(self, hyperparameter, value):
        super().__init__(hyperparameter)
        self.value = value

class ForbiddenEqualsClause(AbstractForbiddenValueClause):
    pass

class ForbiddenInClause(AbstractForbiddenValueClause):
    def __init__(self, hyperparameter, values):
        super().__init__(hyperparameter, values)
        self.values = values

class ForbiddenRelation(AbstractForbiddenClause):
    def __init__(self, hyperparameter, other_hyperparameter):
        super().__init__(hyperparameter)
        self.other_hyperparameter = other_hyperparameter

class AbstractForbiddenConjunction(AbstractForbiddenClause):
    def __init__(self, *args):
        self.components = args

class ForbiddenAndConjunction(AbstractForbiddenConjunction):
    pass
