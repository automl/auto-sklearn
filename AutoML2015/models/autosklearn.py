from AutoSklearn.autosklearn import AutoSklearnClassifier
from AutoSklearn.autosklearn_regression import AutoSklearnRegressor


def get_configuration_space(info, include_classifiers=None,
                            include_preprocessors=None):
    if info['task'] == 'regression':
        if info['is_sparse'] == 1:
            sparse = True
        configuration_space = AutoSklearnRegressor. \
        get_hyperparameter_search_space(sparse=sparse,
                                        exclude_regressors=None)
        return configuration_space
    else:
        task_type = info['task']

        multilabel = False
        multiclass = False
        sparse = False

        if task_type.lower() == 'multilabel.classification':
            multilabel = True
        if task_type.lower() == 'regression':
            raise NotImplementedError()
        if task_type.lower() == 'multiclass.classification':
            multiclass = True
            pass
        if task_type.lower() == 'binary.classification':
            pass

        if info['is_sparse'] == 1:
            sparse = True

        # Todo add a check here if this is useful...
        exclude_classifiers = None
        if include_classifiers is None:
            if sparse == True:
                exclude_classifiers = []
                exclude_classifiers.append('libsvm_svc')

        configuration_space = AutoSklearnClassifier. \
            get_hyperparameter_search_space(multiclass=multiclass,
                                            multilabel=multilabel,
                                            sparse=sparse,
                                            exclude_classifiers=exclude_classifiers,
                                            include_classifiers=include_classifiers,
                                            include_preprocessors=include_preprocessors)
        return configuration_space


def get_model(configuration, seed):
    if 'classifier' in configuration:
        return AutoSklearnClassifier(configuration, seed)
    else:
        raise NotImplementedError()