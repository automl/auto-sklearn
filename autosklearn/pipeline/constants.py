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

DENSE = 'dense'
SPARSE = 'sparse'
PREDICTIONS = 'predictions'
INPUT = 'input'

UNSIGNED_DATA = 'unsigned data' # only +, subset of SIGNED_DATA
SIGNED_DATA = 'signed data' # + or -

DATASET_PROPERTIES_TO_STRING = \
    {DENSE:         'dense',
     SPARSE:        'sparse',
     PREDICTIONS:   'predictions',
     INPUT:         'input',
     UNSIGNED_DATA: 'signed data',
     SIGNED_DATA: 'unsigned data'}