from factories import Metric, MetricFactory


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
        loss_metric = self.factory.create(loss, None)
        metric = LossMetricDecorator(loss_metric)

        return metric