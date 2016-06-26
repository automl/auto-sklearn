from .factories import Metric, MetricFactory


class LossMetricDecorator(Metric):
    '''
    A wrapper object that inverts loss value in order to maximize it instead of
    minimization
    '''

    def __init__(self, loss_metric):
        self.loss_metric = loss_metric

    @property
    def name(self):
        return "reverted_" + self.loss_metric.name

    def calculate_score(self, solution, prediction):
        loss_score = self.loss_metric.calculate_score(solution, prediction)
        metric_score = 1 - loss_score

        return metric_score


class MetricFromLossFactory():

    def __init__(self):
        self.factory = MetricFactory()

    def create(self, loss):
        """
        Function returns a metric object decorated by a loss
        decorator. Scoring function of this metric is a
        function 1 - loss. So, the maximization of this metric
        corresponds to minimization of loss function.
        Parameters
        ----------
        loss : str or callable
        An initial metric to be decorated is created based
        on this parameter. Can be a callable object or
        a package path string.
        """
        loss_metric = self.factory.create(loss, None)
        metric = LossMetricDecorator(loss_metric)

        return metric