import copy
import json

import numpy as np
from smac.tae.execute_ta_run import TAEAbortException
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, KFold, \
    StratifiedKFold, train_test_split, BaseCrossValidator, PredefinedSplit
from sklearn.model_selection._split import _RepeatedSplits, BaseShuffleSplit

from autosklearn.evaluation.abstract_evaluator import AbstractEvaluator
from autosklearn.constants import *


__all__ = ['TrainEvaluator', 'eval_holdout', 'eval_iterative_holdout',
           'eval_cv', 'eval_partial_cv', 'eval_partial_cv_iterative']

__baseCrossValidator_defaults__ = {'GroupKFold': {'n_splits': 3},
                                   'KFold': {'n_splits': 3,
                                             'shuffle': False,
                                             'random_state': None},
                                   'LeaveOneGroupOut': {},
                                   'LeavePGroupsOut': {'n_groups': 2},
                                   'LeaveOneOut': {},
                                   'LeavePOut': {'p': 2},
                                   'PredefinedSplit': {},
                                   'RepeatedKFold': {'n_splits': 5,
                                                     'n_repeats': 10,
                                                     'random_state': None},
                                   'RepeatedStratifiedKFold': {'n_splits': 5,
                                                               'n_repeats': 10,
                                                               'random_state': None},
                                   'StratifiedKFold': {'n_splits': 3,
                                                       'shuffle': False,
                                                       'random_state': None},
                                   'TimeSeriesSplit': {'n_splits': 3,
                                                       'max_train_size': None},
                                   'GroupShuffleSplit': {'n_splits': 5,
                                                         'test_size': 'default',
                                                         'random_state': None},
                                   'StratifiedShuffleSplit': {'n_splits': 10,
                                                              'test_size': 'default',
                                                              'random_state': None},
                                   'ShuffleSplit': {'n_splits': 10,
                                                    'test_size': 'default',
                                                    'random_state': None}
                                   }

def _get_y_array(y, task_type):
    if task_type in CLASSIFICATION_TASKS and task_type != \
            MULTILABEL_CLASSIFICATION:
        return y.ravel()
    else:
        return y



class TrainEvaluator(AbstractEvaluator):
    def __init__(self, backend, queue, metric,
                 configuration=None,
                 all_scoring_functions=False,
                 seed=1,
                 output_y_hat_optimization=True,
                 resampling_strategy=None,
                 resampling_strategy_args=None,
                 num_run=None,
                 subsample=None,
                 keep_models=False,
                 include=None,
                 exclude=None,
                 disable_file_output=False,
                 init_params=None):
        super().__init__(
            backend=backend,
            queue=queue,
            configuration=configuration,
            metric=metric,
            all_scoring_functions=all_scoring_functions,
            seed=seed,
            output_y_hat_optimization=output_y_hat_optimization,
            num_run=num_run,
            subsample=subsample,
            include=include,
            exclude=exclude,
            disable_file_output=disable_file_output,
            init_params=init_params,
        )

        self.resampling_strategy = resampling_strategy
        if resampling_strategy_args is None:
            self.resampling_strategy_args = {}
        else:
            self.resampling_strategy_args = resampling_strategy_args
        self.cv = self.get_splitter(self.datamanager)
        self.cv_folds = self.cv.get_n_splits(
            groups=self.resampling_strategy_args.get('groups')
        )
        self.X_train = self.datamanager.data['X_train']
        self.Y_train = self.datamanager.data['Y_train']
        self.Y_optimization = None
        self.Y_targets = [None] * self.cv_folds
        self.Y_train_targets = np.ones(self.Y_train.shape) * np.NaN
        self.models = [None] * self.cv_folds
        self.indices = [None] * self.cv_folds

        # Necessary for full CV. Makes full CV not write predictions if only
        # a subset of folds is evaluated but time is up. Complicated, because
        #  code must also work for partial CV, where we want exactly the
        # opposite.
        self.partial = True
        self.keep_models = keep_models

    def fit_predict_and_loss(self, iterative=False):
        if iterative:
            if self.cv_folds > 1:
                raise ValueError('Cannot use partial fitting together with full'
                                 'cross-validation!')

            for train_split, test_split in self.cv.split(
                self.X_train, self.Y_train,
                groups=self.resampling_strategy_args.get('groups')
            ):
                self.Y_optimization = self.Y_train[test_split]
                self.Y_actual_train = self.Y_train[train_split]
                self._partial_fit_and_predict(0, train_indices=train_split,
                                              test_indices=test_split,
                                              iterative=True)

        else:

            self.partial = False

            Y_train_pred = [None] * self.cv_folds
            Y_optimization_pred = [None] * self.cv_folds
            Y_valid_pred = [None] * self.cv_folds
            Y_test_pred = [None] * self.cv_folds
            additional_run_info = None
            train_splits = [None] * self.cv_folds

            y = _get_y_array(self.Y_train, self.task_type)
            # TODO: mention that no additional run info is possible in this
            # case! -> maybe remove full CV from the train evaluator anyway and
            # make the user implement this!
            for i, (train_split, test_split) in enumerate(self.cv.split(
                    self.X_train, y,
                    groups=self.resampling_strategy_args.get('groups')
            )):

                # TODO add check that split is actually an integer array,
                # not a boolean array (to allow indexed assignement of
                # training data later).

                (
                    train_pred,
                    opt_pred,
                    valid_pred,
                    test_pred,
                    additional_run_info,
                )= (
                    self._partial_fit_and_predict(
                       i, train_indices=train_split, test_indices=test_split
                    )
                )
                assert len(opt_pred) == len(test_split), (len(opt_pred), len(test_split))

                if (
                    additional_run_info is not None
                    and len(additional_run_info) > 0
                    and i > 0
                ):
                    raise TAEAbortException(
                        'Found additional run info "%s" in fold %d, '
                        'but cannot handle additional run info if fold >= 1.' %
                        (additional_run_info, i)
                    )

                Y_train_pred[i] = train_pred
                Y_optimization_pred[i] = opt_pred
                Y_valid_pred[i] = valid_pred
                Y_test_pred[i] = test_pred
                train_splits[i] = train_split

            Y_targets = self.Y_targets
            Y_train_targets = self.Y_train_targets

            Y_train_pred_full = np.array(
                [
                    np.ones(
                        (self.Y_train.shape[0], Y_train_pred[i].shape[1])
                    ) * np.NaN
                    for _ in range(self.cv_folds) if Y_train_pred[i] is not None
                 ]
            )
            for i in range(self.cv_folds):
                if Y_train_pred[i] is None:
                    continue
                Y_train_pred_full[i][train_splits[i]] = Y_train_pred[i]
            Y_train_pred = np.nanmean(Y_train_pred_full, axis=0)
            if self.cv_folds == 1:
                Y_train_pred = Y_train_pred[
                    # if the first column is np.NaN, all other columns have
                    # to be np.NaN as well
                    np.isfinite(Y_train_pred[:, 0])
                ]


            Y_optimization_pred = np.concatenate(
                [Y_optimization_pred[i] for i in range(self.cv_folds)
                 if Y_optimization_pred[i] is not None])
            Y_targets = np.concatenate([Y_targets[i] for i in range(self.cv_folds)
                                        if Y_targets[i] is not None])

            if self.X_valid is not None:
                Y_valid_pred = np.array([Y_valid_pred[i]
                                         for i in range(self.cv_folds)
                                         if Y_valid_pred[i] is not None])
                # Average the predictions of several models
                if len(Y_valid_pred.shape) == 3:
                    Y_valid_pred = np.nanmean(Y_valid_pred, axis=0)
            else:
                Y_valid_pred = None

            if self.X_test is not None:
                Y_test_pred = np.array([Y_test_pred[i]
                                        for i in range(self.cv_folds)
                                        if Y_test_pred[i] is not None])
                # Average the predictions of several models
                if len(Y_test_pred.shape) == 3:
                    Y_test_pred = np.nanmean(Y_test_pred, axis=0)
            else:
                Y_test_pred = None

            self.Y_optimization = Y_targets
            loss = self._loss(Y_targets, Y_optimization_pred)
            self.Y_actual_train = Y_train_targets

            if self.cv_folds > 1:
                self.model = self._get_model()
                # Bad style, but necessary for unit testing that self.model is
                # actually a new model
                self._added_empty_model = True

            self.finish_up(
                loss=loss,
                train_pred=Y_train_pred,
                opt_pred=Y_optimization_pred,
                valid_pred=Y_valid_pred,
                test_pred=Y_test_pred,
                additional_run_info=additional_run_info,
                file_output=True,
                final_call=True
            )

    def partial_fit_predict_and_loss(self, fold, iterative=False):
        if fold > self.cv_folds:
            raise ValueError('Cannot evaluate a fold %d which is higher than '
                             'the number of folds %d.' % (fold, self.cv_folds))

        y = _get_y_array(self.Y_train, self.task_type)
        for i, (train_split, test_split) in enumerate(self.cv.split(
                self.X_train, y,
                groups=self.resampling_strategy_args.get('groups')
        )):
            if i != fold:
                continue
            else:
                break

        if self.cv_folds > 1:
            self.Y_optimization = self.Y_train[test_split]
            self.Y_actual_train = self.Y_train[train_split]

        if iterative:
            self._partial_fit_and_predict(
                fold, train_indices=train_split, test_indices=test_split,
                iterative=iterative)
        else:
            train_pred, opt_pred, valid_pred, test_pred, additional_run_info = (
                self._partial_fit_and_predict(
                    fold,
                    train_indices=train_split,
                    test_indices=test_split,
                    iterative=iterative,
                )
            )
            loss = self._loss(self.Y_targets[fold], opt_pred)

            if self.cv_folds > 1:
                self.model = self._get_model()
                # Bad style, but necessary for unit testing that self.model is
                # actually a new model
                self._added_empty_model = True

            self.finish_up(
                loss=loss,
                train_pred=train_pred,
                opt_pred=opt_pred,
                valid_pred=valid_pred,
                test_pred=test_pred,
                file_output=False,
                final_call=True,
                additional_run_info=None,
            )

    def _partial_fit_and_predict(self, fold, train_indices, test_indices,
                                 iterative=False):
        model = self._get_model()

        train_indices = self.subsample_indices(train_indices)

        self.indices[fold] = ((train_indices, test_indices))

        if iterative:

            # Do only output the files in the case of iterative holdout,
            # In case of iterative partial cv, no file output is needed
            # because ensembles cannot be built
            file_output = True if self.cv_folds == 1 else False

            if model.estimator_supports_iterative_fit():
                Xt, fit_params = model.fit_transformer(self.X_train[train_indices],
                                                       self.Y_train[train_indices])

                self.Y_train_targets[train_indices] = self.Y_train[train_indices]

                iteration = 1
                total_n_iteration = 0
                while (
                    not model.configuration_fully_fitted()
                ):
                    n_iter = int(2**iteration/2) if iteration > 1 else 2
                    total_n_iteration += n_iter
                    model.iterative_fit(Xt, self.Y_train[train_indices],
                                        n_iter=n_iter, **fit_params)
                    (
                        Y_train_pred,
                        Y_optimization_pred,
                        Y_valid_pred,
                        Y_test_pred
                    ) = self._predict(
                        model,
                        train_indices=train_indices,
                        test_indices=test_indices,
                    )

                    if self.cv_folds == 1:
                        self.model = model

                    loss = self._loss(self.Y_train[test_indices], Y_optimization_pred)
                    additional_run_info = model.get_additional_run_info()

                    if model.configuration_fully_fitted():
                        final_call = True
                    else:
                        final_call = False
                    self.finish_up(
                        loss=loss,
                        train_pred=Y_train_pred,
                        opt_pred=Y_optimization_pred,
                        valid_pred=Y_valid_pred,
                        test_pred=Y_test_pred,
                        additional_run_info=additional_run_info,
                        file_output=file_output,
                        final_call=final_call,
                    )
                    iteration += 1

                return
            else:
                self._fit_and_suppress_warnings(model,
                                                self.X_train[train_indices],
                                                self.Y_train[train_indices])

                if self.cv_folds == 1:
                    self.model = model

                train_indices, test_indices = self.indices[fold]
                self.Y_targets[fold] = self.Y_train[test_indices]
                self.Y_train_targets[train_indices] = self.Y_train[train_indices]
                (
                    Y_train_pred,
                    Y_optimization_pred,
                    Y_valid_pred,
                    Y_test_pred
                ) = self._predict(
                    model=model,
                    train_indices=train_indices,
                    test_indices=test_indices
                )
                loss = self._loss(self.Y_train[test_indices], Y_optimization_pred)
                additional_run_info = model.get_additional_run_info()
                self.finish_up(
                    loss=loss,
                    train_pred=Y_train_pred,
                    opt_pred=Y_optimization_pred,
                    valid_pred=Y_valid_pred,
                    test_pred=Y_test_pred,
                    additional_run_info=additional_run_info,
                    file_output=file_output,
                    final_call=True
                )
                return

        else:
            self._fit_and_suppress_warnings(model,
                                            self.X_train[train_indices],
                                            self.Y_train[train_indices])

            if self.cv_folds == 1:
                self.model = model

            train_indices, test_indices = self.indices[fold]
            self.Y_targets[fold] = self.Y_train[test_indices]
            self.Y_train_targets[train_indices] = self.Y_train[train_indices]

            train_pred, opt_pred, valid_pred, test_pred = self._predict(
                model=model,
                train_indices=train_indices,
                test_indices=test_indices,
            )
            additional_run_info = model.get_additional_run_info()
            return (
                train_pred,
                opt_pred,
                valid_pred,
                test_pred,
                additional_run_info,
            )

    def subsample_indices(self, train_indices):
        if self.subsample is not None:
            # Only subsample if there are more indices given to this method than
            # required to subsample because otherwise scikit-learn will complain

            if self.task_type in CLASSIFICATION_TASKS and \
                    self.task_type != MULTILABEL_CLASSIFICATION:
                stratify = self.Y_train[train_indices]
            else:
                stratify = None

            if len(train_indices) > self.subsample:
                indices = np.arange(len(train_indices))
                cv_indices_train, _ = train_test_split(
                    indices,
                    stratify=stratify,
                    train_size=self.subsample,
                    random_state=1,
                    shuffle=True,
                )
                train_indices = train_indices[cv_indices_train]
                return train_indices

        return train_indices

    def _predict(self, model, test_indices, train_indices):
        train_pred = self.predict_function(self.X_train[train_indices],
                                           model, self.task_type,
                                           self.Y_train[train_indices])

        opt_pred = self.predict_function(self.X_train[test_indices],
                                         model, self.task_type,
                                         self.Y_train[train_indices])

        if self.X_valid is not None:
            X_valid = self.X_valid.copy()
            valid_pred = self.predict_function(X_valid, model,
                                               self.task_type,
                                               self.Y_train[train_indices])
        else:
            valid_pred = None

        if self.X_test is not None:
            X_test = self.X_test.copy()
            test_pred = self.predict_function(X_test, model,
                                              self.task_type,
                                              self.Y_train[train_indices])
        else:
            test_pred = None

        return train_pred, opt_pred, valid_pred, test_pred

    def get_splitter(self, D):

        if self.resampling_strategy_args is None:
            self.resampling_strategy_args = {}

        if not isinstance(self.resampling_strategy, str):

            if issubclass(self.resampling_strategy, BaseCrossValidator) or\
                issubclass(self.resampling_strategy, _RepeatedSplits) or\
                issubclass(self.resampling_strategy, BaseShuffleSplit):

                class_name = self.resampling_strategy.__name__
                if class_name not in __baseCrossValidator_defaults__:
                    raise ValueError('Unknown CrossValidator.')
                ref_arg_dict = __baseCrossValidator_defaults__[class_name]

                y = D.data['Y_train'].ravel()
                if class_name == 'PredefinedSplit':
                    if 'test_fold' not in self.resampling_strategy_args:
                        raise ValueError('Must provide parameter test_fold'
                                         ' for class PredefinedSplit.')
                if class_name == 'LeaveOneGroupOut' or \
                        class_name == 'LeavePGroupsOut' or\
                        class_name == 'GroupKFold' or\
                        class_name == 'GroupShuffleSplit':
                    if 'groups' not in self.resampling_strategy_args:
                        raise ValueError('Must provide parameter groups '
                                         'for chosen CrossValidator.')
                    try:
                        if self.resampling_strategy_args['groups'].shape != y.shape:
                            raise ValueError('Groups must be array-like '
                                             'with shape (n_samples,).')
                    except Exception:
                        raise ValueError('Groups must be array-like '
                                         'with shape (n_samples,).')
                else:
                    if 'groups' in self.resampling_strategy_args:
                        if self.resampling_strategy_args['groups'].shape != y.shape:
                            raise ValueError('Groups must be array-like'
                                             ' with shape (n_samples,).')

                # Put args in self.resampling_strategy_args
                for key in ref_arg_dict:
                    if key == 'n_splits':
                        if 'folds' not in self.resampling_strategy_args:
                            self.resampling_strategy_args['folds'] = ref_arg_dict['n_splits']
                    else:
                        if key not in self.resampling_strategy_args:
                            self.resampling_strategy_args[key] = ref_arg_dict[key]

                # Instantiate object with args
                init_dict = copy.deepcopy(self.resampling_strategy_args)
                init_dict.pop('groups', None)
                if 'folds' in init_dict:
                    init_dict['n_splits'] = init_dict.pop('folds', None)
                cv = copy.deepcopy(self.resampling_strategy)(**init_dict)

                if 'groups' not in self.resampling_strategy_args:
                    self.resampling_strategy_args['groups'] = None

                return cv

        y = D.data['Y_train']
        shuffle = self.resampling_strategy_args.get('shuffle', True)
        train_size = 0.67
        if self.resampling_strategy_args:
            train_size = self.resampling_strategy_args.get('train_size',
                                                           train_size)
        test_size = float("%.4f" % (1 - train_size))

        if D.info['task'] in CLASSIFICATION_TASKS and \
                        D.info['task'] != MULTILABEL_CLASSIFICATION:

            y = y.ravel()
            if self.resampling_strategy in ['holdout',
                                            'holdout-iterative-fit']:

                if shuffle:
                    try:
                        cv = StratifiedShuffleSplit(n_splits=1,
                                                    test_size=test_size,
                                                    random_state=1)
                        test_cv = copy.deepcopy(cv)
                        next(test_cv.split(y, y))
                    except ValueError as e:
                        if 'The least populated class in y has only' in e.args[0]:
                            cv = ShuffleSplit(n_splits=1, test_size=test_size,
                                              random_state=1)
                        else:
                            raise e
                else:
                    tmp_train_size = int(np.floor(train_size * y.shape[0]))
                    test_fold = np.zeros(y.shape[0])
                    test_fold[:tmp_train_size] = -1
                    cv = PredefinedSplit(test_fold=test_fold)
                    cv.n_splits = 1  # As sklearn is inconsistent here
            elif self.resampling_strategy in ['cv', 'partial-cv',
                                              'partial-cv-iterative-fit']:
                if shuffle:
                    cv = StratifiedKFold(
                        n_splits=self.resampling_strategy_args['folds'],
                        shuffle=shuffle, random_state=1)
                else:
                    cv = KFold(n_splits=self.resampling_strategy_args['folds'],
                               shuffle=shuffle, random_state=1)
            else:
                raise ValueError(self.resampling_strategy)
        else:
            if self.resampling_strategy in ['holdout',
                                            'holdout-iterative-fit']:
                # TODO shuffle not taken into account for this
                if shuffle:
                    cv = ShuffleSplit(n_splits=1, test_size=test_size,
                                      random_state=1)
                else:
                    tmp_train_size = int(np.floor(train_size * y.shape[0]))
                    test_fold = np.zeros(y.shape[0])
                    test_fold[:tmp_train_size] = -1
                    cv = PredefinedSplit(test_fold=test_fold)
                    cv.n_splits = 1  # As sklearn is inconsistent here
            elif self.resampling_strategy in ['cv', 'partial-cv',
                                              'partial-cv-iterative-fit']:
                cv = KFold(n_splits=self.resampling_strategy_args['folds'],
                           shuffle=shuffle, random_state=1)
            else:
                raise ValueError(self.resampling_strategy)
        return cv


# create closure for evaluating an algorithm
def eval_holdout(
        queue,
        config,
        backend,
        resampling_strategy,
        resampling_strategy_args,
        metric,
        seed,
        num_run,
        instance,
        all_scoring_functions,
        output_y_hat_optimization,
        include,
        exclude,
        disable_file_output,
        init_params=None,
        iterative=False,
):
    instance = json.loads(instance) if instance is not None else {}
    subsample = instance.get('subsample')
    evaluator = TrainEvaluator(
        backend=backend,
        queue=queue,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        metric=metric,
        configuration=config,
        seed=seed,
        num_run=num_run,
        subsample=subsample,
        all_scoring_functions=all_scoring_functions,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        init_params=init_params
    )
    evaluator.fit_predict_and_loss(iterative=iterative)


def eval_iterative_holdout(
        queue,
        config,
        backend,
        resampling_strategy,
        resampling_strategy_args,
        metric,
        seed,
        num_run,
        instance,
        all_scoring_functions,
        output_y_hat_optimization,
        include,
        exclude,
        disable_file_output,
        init_params=None,
):
    return eval_holdout(
        queue=queue,
        config=config,
        backend=backend,
        metric=metric,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        seed=seed,
        num_run=num_run,
        all_scoring_functions=all_scoring_functions,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        instance=instance,
        disable_file_output=disable_file_output,
        iterative=True,
        init_params=init_params
    )


def eval_partial_cv(
        queue,
        config,
        backend,
        resampling_strategy,
        resampling_strategy_args,
        metric,
        seed,
        num_run,
        instance,
        all_scoring_functions,
        output_y_hat_optimization,
        include,
        exclude,
        disable_file_output,
        init_params=None,
        iterative=False
):
    instance = json.loads(instance) if instance is not None else {}
    subsample = instance.get('subsample')
    fold = instance['fold']

    evaluator = TrainEvaluator(
        backend=backend,
        queue=queue,
        metric=metric,
        configuration=config,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        seed=seed,
        num_run=num_run,
        subsample=subsample,
        all_scoring_functions=all_scoring_functions,
        output_y_hat_optimization=False,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        init_params=init_params,
    )

    evaluator.partial_fit_predict_and_loss(fold=fold, iterative=iterative)


def eval_partial_cv_iterative(
        queue,
        config,
        backend,
        resampling_strategy,
        resampling_strategy_args,
        metric,
        seed,
        num_run,
        instance,
        all_scoring_functions,
        output_y_hat_optimization,
        include,
        exclude,
        disable_file_output,
        init_params=None,
):
    return eval_partial_cv(
        queue=queue,
        config=config,
        backend=backend,
        metric=metric,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        seed=seed,
        num_run=num_run,
        instance=instance,
        all_scoring_functions=all_scoring_functions,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        iterative=True,
        init_params=init_params,
    )


# create closure for evaluating an algorithm
def eval_cv(
        queue,
        config,
        backend,
        resampling_strategy,
        resampling_strategy_args,
        metric,
        seed,
        num_run,
        instance,
        all_scoring_functions,
        output_y_hat_optimization,
        include,
        exclude,
        disable_file_output,
        init_params=None,
):
    instance = json.loads(instance) if instance is not None else {}
    subsample = instance.get('subsample')
    evaluator = TrainEvaluator(
        backend=backend,
        queue=queue,
        metric=metric,
        configuration=config,
        seed=seed,
        num_run=num_run,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        subsample=subsample,
        all_scoring_functions=all_scoring_functions,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        init_params=init_params,
    )

    evaluator.fit_predict_and_loss()
