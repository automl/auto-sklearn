from AutoSklearn.autosklearn import AutoSklearnClassifier


def get_configuration_space(info):
    if info['task'] == 'regression':
        raise NotImplementedError()
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

        configuration_space = AutoSklearnClassifier. \
            get_hyperparameter_search_space(multiclass=multiclass,
                                            multilabel=multilabel,
                                            sparse=sparse)
        return configuration_space


def get_model(configuration, seed):
    if 'classifier' in configuration:
        return AutoSklearnClassifier(configuration, seed)
    else:
        raise NotImplementedError()