# -*- encoding: utf-8 -*-
BINARY_CLASSIFICATION = 1
MULTICLASS_CLASSIFICATION = 2
MULTILABEL_CLASSIFICATION = 3
REGRESSION = 4

REGRESSION_TASKS = [REGRESSION]
CLASSIFICATION_TASKS = [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION,
                        MULTILABEL_CLASSIFICATION]

TASK_TYPES = REGRESSION_TASKS + CLASSIFICATION_TASKS

TASK_TYPES_TO_STRING = \
    {BINARY_CLASSIFICATION: 'binary.classification',
     MULTICLASS_CLASSIFICATION: 'multiclass.classification',
     MULTILABEL_CLASSIFICATION: 'multilabel.classification',
     REGRESSION: 'regression'}

STRING_TO_TASK_TYPES = \
    {'binary.classification': BINARY_CLASSIFICATION,
     'multiclass.classification': MULTICLASS_CLASSIFICATION,
     'multilabel.classification': MULTILABEL_CLASSIFICATION,
     'regression': REGRESSION}

ACC_METRIC = 5
AUC_METRIC = 6
BAC_METRIC = 7
F1_METRIC = 8
PAC_METRIC = 9
CLASSIFICATION_METRICS = [ACC_METRIC, AUC_METRIC, BAC_METRIC,
                          F1_METRIC, PAC_METRIC]

R2_METRIC = 10
A_METRIC = 11
REGRESSION_METRICS = [R2_METRIC, A_METRIC]
METRIC = CLASSIFICATION_METRICS + REGRESSION_METRICS
STRING_TO_METRIC = {
    'acc': ACC_METRIC,
    'acc_metric': ACC_METRIC,
    'auc': AUC_METRIC,
    'auc_metric': AUC_METRIC,
    'bac': BAC_METRIC,
    'bac_metric': BAC_METRIC,
    'f1': F1_METRIC,
    'f1_metric': F1_METRIC,
    'pac': PAC_METRIC,
    'pac_metric': PAC_METRIC,
    'r2': R2_METRIC,
    'r2_metric': R2_METRIC,
    'a': A_METRIC,
    'a_metric': A_METRIC
}

METRIC_TO_STRING = {
    ACC_METRIC: 'acc_metric',
    AUC_METRIC: 'auc_metric',
    BAC_METRIC: 'bac_metric',
    F1_METRIC: 'f1_metric',
    PAC_METRIC: 'pac_metric',
    R2_METRIC: 'r2_metric',
    A_METRIC: 'a_metric'
}

METRICS_SHORT_TO_LONG_FORM = {
    'acc': 'acc_metric',
    'auc': 'auc_metric',
    'bac': 'bac_metric',
    'f1': 'f1_metric',
    'pac': 'pac_metric',
    'r2': 'r2_metric',
    'a': 'a_metric'
}

