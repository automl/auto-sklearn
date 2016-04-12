import six
import importlib
from autosklearn.constants import BINARY_CLASSIFICATION, \
    MULTICLASS_CLASSIFICATION, MULTILABEL_CLASSIFICATION, REGRESSION
from classification_metrics import acc_metric, bac_metric, pac_metric, \
    f1_metric, auc_metric
from regression_metrics import a_metric, r2_metric


### Metric base class and implementations ###

class Metric():

    name = None

    def calculate_score(self, solution, prediction):
        pass


class KnownMetric(Metric):

    def __init__(self, name, scoring_function, task_type):
        self.name = name
        self.task_type = task_type
        self._scoring_function = scoring_function

    def calculate_score(self, solution, prediction):
        return self._scoring_function(solution, prediction, self.task_type)


class CustomMetric(Metric):

    name = 'custom'

    def __init__(self, scoring_function):
        self._scoring_function = scoring_function

    def calculate_score(self, solution, prediction):
        return self._scoring_function(solution, prediction)


class PackageMetric(CustomMetric):

    def __init__(self, package, scoring_function):
        self.name = package
        CustomMetric.__init__(self, scoring_function)

###


STRING_TO_METRIC = {
    'acc': acc_metric,
    'acc_metric': acc_metric,
    'auc': auc_metric,
    'auc_metric': auc_metric,
    'bac': bac_metric,
    'bac_metric': bac_metric,
    'f1': f1_metric,
    'f1_metric': f1_metric,
    'pac': pac_metric,
    'pac_metric': pac_metric,
    'r2': r2_metric,
    'r2_metric': r2_metric,
    'a': a_metric,
    'a_metric': a_metric
}


class MetricBuilder():

    metric_name = None
    scoring_function = None
    possible_metric_names = None
    possible_task_types = None

    def can_build_metric_with_name(self, metric_name):
        return metric_name in self.possible_metric_names

    def can_build(self, task_type):
        if self.possible_task_types:
            return task_type in self.possible_task_types
        else:
            return True

    def build(self, task_type):
        if self.can_build(task_type):
            return KnownMetric(self.metric_name,
                               self.scoring_function,
                               task_type)
        else:
            raise ValueError('Task type %s is not supported by the builder.'
                             % task_type)

    def get_scoring_function(self):
        pass

'''
    Metric builders specify:
        - the name of the metric they build
        - scoring function of the metric
        - possible names, by which the metric is accessible (aliases)
        - possible task types, for which this metric can be returned
'''

class AMetricBuilder(MetricBuilder):
    metric_name = 'a_metric'
    scoring_function = staticmethod(a_metric)
    possible_metric_names = ['a_metric', 'a']
    possible_task_types = [REGRESSION]


class R2MetricBuilder(MetricBuilder):
    metric_name = 'r2_metric'
    scoring_function = staticmethod(r2_metric)
    possible_metric_names = ['r2_metric', 'r2']
    possible_task_types = [REGRESSION]


class ACCMetricBuilder(MetricBuilder):
    metric_name = 'acc_metric'
    scoring_function = staticmethod(acc_metric)
    possible_metric_names = ['acc_metric', 'acc']
    possible_task_types = [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION,
                           MULTILABEL_CLASSIFICATION]


class BACMetricBuilder(MetricBuilder):
    metric_name = 'bac_metric'
    scoring_function = staticmethod(bac_metric)
    possible_metric_names = ['bac_metric', 'bac']
    possible_task_types = [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION,
                           MULTILABEL_CLASSIFICATION]


class PACMetricBuilder(MetricBuilder):
    metric_name = 'pac_metric'
    scoring_function = staticmethod(pac_metric)
    possible_metric_names = ['pac_metric', 'pac']
    possible_task_types = [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION,
                           MULTILABEL_CLASSIFICATION]


class F1MetricBuilder(MetricBuilder):
    metric_name = 'f1_metric'
    scoring_function = staticmethod(f1_metric)
    possible_metric_names = ['f1_metric', 'f1']
    possible_task_types = [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION,
                           MULTILABEL_CLASSIFICATION]


class AUCMetricBuilder(MetricBuilder):
    metric_name = 'auc_metric'
    scoring_function = staticmethod(auc_metric)
    possible_metric_names = ['auc_metric', 'auc']
    possible_task_types = [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION,
                           MULTILABEL_CLASSIFICATION]


class KnownMetricFactory():

    all_builders = [F1MetricBuilder(),
                    PACMetricBuilder(),
                    ACCMetricBuilder(),
                    AUCMetricBuilder(),
                    AMetricBuilder(),
                    BACMetricBuilder(),
                    R2MetricBuilder()]

    def __init__(self, task_type):
        self.task_type = task_type
        self.builders_for_task_type = [b for b in self.all_builders
                                       if b.can_build(task_type)]

    def create(self, metric_name):
        builders = [b for b in self.builders_for_task_type
                    if b.can_build_metric_with_name(metric_name)]
        n_builders = len(builders)

        if n_builders == 1:
            builder = builders[0]
            metric = builder.build(self.task_type)

            return metric
        elif n_builders == 0:
            raise ValueError('No metric builder to metric with name %s.'
                             % metric_name)
        else:
            raise ValueError('Several builders found to metric with name %s.'
                             % metric_name)

    def get_all(self):
        for builder in self.builders_for_task_type:
            yield builder.build(self.task_type)


class PackageMetricFactory():

    def create(self, package):
        last_dot_index = package.rfind('.')
        module_name = package[:last_dot_index]
        function_name = package[last_dot_index + 1:]

        module = importlib.import_module(module_name)
        scoring_function = getattr(module, function_name)

        if not callable(scoring_function):
            raise Exception('Package %s endpoint is not a function' % package)

        return PackageMetric(package, scoring_function)


class MetricFactory():

    def create(self, metric, task_type=None):
        if isinstance(metric, Metric):
            return metric
        elif callable(metric):
            return CustomMetric(metric)
        elif isinstance(metric, six.string_types):
            if '.' in metric:
                package_metric_factory = PackageMetricFactory()
                metric = package_metric_factory.create(metric)
                return metric
            else:
                known_metric_factory = KnownMetricFactory(task_type)
                metric = known_metric_factory.create(metric_name=metric)
                return metric
