import numpy as np
import sklearn.cross_validation

from autosklearn.evaluation.abstract_evaluator import AbstractEvaluator
from autosklearn.constants import *


__all__ = ['TrainEvaluator', 'eval_holdout', 'eval_iterative_holdout',
           'eval_cv', 'eval_partial_cv', 'eval_partial_cv_iterative']


class TrainEvaluator(AbstractEvaluator):
    def __init__(self, Datamanager, backend, queue,
                 configuration=None,
                 with_predictions=False,
                 all_scoring_functions=False,
                 seed=1,
                 output_y_test=False,
                 cv=None,
                 num_run=None,
                 subsample=None,
                 keep_models=False,
                 include=None,
                 exclude=None,
                 disable_file_output=False):
        super().__init__(
            Datamanager, backend, queue,
            configuration=configuration,
            with_predictions=with_predictions,
            all_scoring_functions=all_scoring_functions,
            seed=seed,
            output_y_test=output_y_test,
            num_run=num_run,
            subsample=subsample,
            include=include,
            exclude=exclude,
            disable_file_output=disable_file_output)

        self.cv = cv
        self.cv_folds = cv.n_folds if hasattr(cv, 'n_folds') else cv.n_iter
        self.X_train = self.D.data['X_train']
        self.Y_train = self.D.data['Y_train']
        self.Y_optimization = None
        self.Y_targets = [None] * self.cv_folds
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

            for train_split, test_split in self.cv:
                self.Y_optimization = self.Y_train[test_split]
                self._partial_fit_and_predict(0, train_indices=train_split,
                                              test_indices=test_split,
                                              iterative=True)

        else:

            self.partial = False

            Y_optimization_pred = [None] * self.cv_folds
            Y_valid_pred = [None] * self.cv_folds
            Y_test_pred = [None] * self.cv_folds

            for i, (train_split, test_split) in enumerate(self.cv):
                opt_pred, valid_pred, test_pred = self._partial_fit_and_predict(
                    i, train_indices=train_split, test_indices=test_split)

                Y_optimization_pred[i] = opt_pred
                Y_valid_pred[i] = valid_pred
                Y_test_pred[i] = test_pred

            Y_targets = self.Y_targets

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

            if self.cv_folds > 1:
                self.model = self._get_model()
                # Bad style, but necessary for unit testing that self.model is
                # actually a new model
                self._added_empty_model = True

            self.finish_up(loss, Y_optimization_pred, Y_valid_pred, Y_test_pred,
                           file_output=True)

    def partial_fit_predict_and_loss(self, fold, iterative=False):
        if fold > self.cv_folds:
            raise ValueError('Cannot evaluate a fold %d which is higher than '
                             'the number of folds %d.' % (fold, self.cv_folds))
        for i, (train_split, test_split) in enumerate(self.cv):
            if i != fold:
                continue
            else:
                break

        if self.cv_folds > 1:
            self.Y_optimization = self.Y_train[test_split]

        if iterative:
            self._partial_fit_and_predict(
                fold, train_indices=train_split, test_indices=test_split,
                iterative=iterative)
        else:
            opt_pred, valid_pred, test_pred = self._partial_fit_and_predict(
                fold, train_indices=train_split, test_indices=test_split,
                iterative=iterative)
            loss = self._loss(self.Y_targets[fold], opt_pred)

            if self.cv_folds > 1:
                self.model = self._get_model()
                # Bad style, but necessary for unit testing that self.model is
                # actually a new model
                self._added_empty_model = True

            self.finish_up(loss, opt_pred, valid_pred, test_pred,
                           file_output=False)

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
                Xt, fit_params = model.pre_transform(self.X_train[train_indices],
                                                     self.Y_train[train_indices])

                n_iter = 2
                while not model.configuration_fully_fitted():
                    model.iterative_fit(Xt, self.Y_train[train_indices],
                                        n_iter=n_iter, **fit_params)
                    Y_optimization_pred, Y_valid_pred, Y_test_pred = self._predict(
                        model, train_indices=train_indices, test_indices=test_indices)

                    if self.cv_folds == 1:
                        self.model = model

                    loss = self._loss(self.Y_train[test_indices], Y_optimization_pred)

                    self.finish_up(loss, Y_optimization_pred, Y_valid_pred,
                                   Y_test_pred, file_output=file_output)
                    n_iter *= 2

                return
            else:
                self._fit_and_suppress_warnings(model,
                                                self.X_train[train_indices],
                                                self.Y_train[train_indices])

                if self.cv_folds == 1:
                    self.model = model

                train_indices, test_indices = self.indices[fold]
                self.Y_targets[fold] = self.Y_train[test_indices]
                Y_optimization_pred, Y_valid_pred, Y_test_pred = self._predict(
                    model=model, train_indices=train_indices, test_indices=test_indices)
                loss = self._loss(self.Y_train[test_indices], Y_optimization_pred)
                self.finish_up(loss, Y_optimization_pred, Y_valid_pred,
                               Y_test_pred, file_output=file_output)
                return

        else:
            self._fit_and_suppress_warnings(model,
                                            self.X_train[train_indices],
                                            self.Y_train[train_indices])

            if self.cv_folds == 1:
                self.model = model

            train_indices, test_indices = self.indices[fold]
            self.Y_targets[fold] = self.Y_train[test_indices]
            return self._predict(model=model, train_indices=train_indices,
                                 test_indices=test_indices)

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
                cv_indices_train, _ = sklearn.cross_validation.train_test_split(
                    indices, stratify=stratify,
                    train_size=self.subsample, random_state=1)
                train_indices = train_indices[cv_indices_train]
                return train_indices

        return train_indices


    def _predict(self, model, test_indices, train_indices):
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

        return opt_pred, valid_pred, test_pred


# create closure for evaluating an algorithm
def eval_holdout(queue, config, data, backend, cv, seed, num_run,
                 subsample, with_predictions, all_scoring_functions,
                 output_y_test, include, exclude, disable_file_output,
                 iterative=False):
    evaluator = TrainEvaluator(data, backend, queue,
                               cv=cv,
                               configuration=config,
                               seed=seed,
                               num_run=num_run,
                               subsample=subsample,
                               with_predictions=with_predictions,
                               all_scoring_functions=all_scoring_functions,
                               output_y_test=output_y_test,
                               include=include,
                               exclude=exclude,
                               disable_file_output=disable_file_output)
    evaluator.fit_predict_and_loss(iterative=iterative)


def eval_iterative_holdout(queue, config, data, backend, cv, seed,
                           num_run, subsample, with_predictions,
                           all_scoring_functions, output_y_test,
                           include, exclude, disable_file_output):
    return eval_holdout(queue=queue, config=config, data=data, backend=backend,
                        cv=cv, seed=seed, num_run=num_run, subsample=subsample,
                        with_predictions=with_predictions,
                        all_scoring_functions=all_scoring_functions,
                        output_y_test=output_y_test,
                        include=include, exclude=exclude,
                        disable_file_output=disable_file_output, iterative=True)


def eval_partial_cv(queue, config, data, backend, cv, seed, num_run, instance,
                    subsample, with_predictions, all_scoring_functions,
                    output_y_test, include, exclude, disable_file_output,
                    iterative=False):
    evaluator = TrainEvaluator(data, backend, queue,
                               configuration=config,
                               cv=cv,
                               seed=seed,
                               num_run=num_run,
                               subsample=subsample,
                               with_predictions=with_predictions,
                               all_scoring_functions=all_scoring_functions,
                               output_y_test=False,
                               include=include,
                               exclude=exclude,
                               disable_file_output=disable_file_output)

    evaluator.partial_fit_predict_and_loss(fold=instance, iterative=iterative)


def eval_partial_cv_iterative(queue, config, data, backend, cv, seed, num_run,
                              instance, subsample, with_predictions,
                              all_scoring_functions, output_y_test,
                              include, exclude, disable_file_output):
    return eval_partial_cv(queue=queue, config=config, data=data, backend=backend,
                           cv=cv, seed=seed, num_run=num_run, instance=instance,
                           subsample=subsample, with_predictions=with_predictions,
                           all_scoring_functions=all_scoring_functions,
                           output_y_test=output_y_test, include=include,
                           exclude=exclude, disable_file_output=disable_file_output,
                           iterative=True)


# create closure for evaluating an algorithm
def eval_cv(queue, config, data, backend, cv, seed, num_run,
            subsample, with_predictions, all_scoring_functions,
            output_y_test, include, exclude, disable_file_output):
    evaluator = TrainEvaluator(data, backend, queue,
                               configuration=config,
                               seed=seed,
                               num_run=num_run,
                               cv=cv,
                               subsample=subsample,
                               with_predictions=with_predictions,
                               all_scoring_functions=all_scoring_functions,
                               output_y_test=output_y_test,
                               include=include,
                               exclude=exclude,
                               disable_file_output=disable_file_output)

    evaluator.fit_predict_and_loss()
