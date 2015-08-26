"""Constants which are used as dataset properties.
"""
BINARY_CLASSIFICATION = 1
MULTICLASS_CLASSIFICATION = 2
MULTILABEL_CLASSIFICATION = 3
REGRESSION = 4

REGRESSION_TASKS = [REGRESSION]
CLASSIFICATION_TASKS = [BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION,
                        MULTILABEL_CLASSIFICATION]

TASK_TYPES = REGRESSION_TASKS + CLASSIFICATION_TASKS

TASK_TYPES_TO_STRING = \
    {BINARY_CLASSIFICATION: "binary.classification",
     MULTICLASS_CLASSIFICATION: "multiclass.classification",
     MULTILABEL_CLASSIFICATION: "multilabel.classification",
     REGRESSION: "regression"}

STRING_TO_TASK_TYPES = \
    {"binary.classification": BINARY_CLASSIFICATION,
     "multiclass.classification": MULTICLASS_CLASSIFICATION,
     "multilabel.classification": MULTILABEL_CLASSIFICATION,
     "regression": REGRESSION}

DENSE = 5
SPARSE = 6
PREDICTIONS = 7
INPUT = 8

SIGNED_DATA = 9
UNSIGNED_DATA = 10

DATASET_PROPERTIES_TO_STRING = \
    {DENSE:         'dense',
     SPARSE:        'sparse',
     PREDICTIONS:   'predictions',
     INPUT:         'input',
     SIGNED_DATA:   'signed data',
     UNSIGNED_DATA: 'unsigned data'}