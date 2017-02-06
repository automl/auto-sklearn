import numpy as np
from smac.tae.execute_ta_run import StatusType

from autosklearn.evaluation.abstract_evaluator import AbstractEvaluator


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
        """Fit function for non-iterative fitting of models"""
        self.partial = False

        if iterative and self.cv_folds > 1:
            raise ValueError('Cannot use partial fitting together with full'
                             'cross-validation!')

        Y_optimization_pred = [None] * self.cv_folds
        Y_valid_pred = [None] * self.cv_folds
        Y_test_pred = [None] * self.cv_folds

        for i, (train_split, test_split) in enumerate(self.cv):
            opt_pred, valid_pred, test_pred = self._partial_fit_and_predict(
                i, train_indices=train_split, test_indices=test_split,
                iterative=iterative)

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

        return loss, Y_optimization_pred, Y_valid_pred, Y_test_pred

    def partial_fit_predict_and_loss(self, fold, iterative=False):
        if fold > self.cv_folds:
            raise ValueError('Cannot evaluate a fold %d which is higher than '
                             'the number of folds %d.' % (fold, self.cv_folds))
        for i, (train_split, test_split) in enumerate(self.cv):
            if i != fold:
                continue
            else:
                break

        opt_pred, valid_pred, test_pred = self._partial_fit_and_predict(
            fold, train_indices=train_split, test_indices=test_split,
            iterative=iterative)
        loss = self._loss(self.Y_targets[fold], opt_pred)

        if self.cv_folds > 1:
            self.model = self._get_model()
            # Bad style, but necessary for unit testing that self.model is
            # actually a new model
            self._added_empty_model = True

        return loss, opt_pred, valid_pred, test_pred

    def _partial_fit_and_predict(self, fold, train_indices, test_indices,
                                 iterative=False):
        model = self._get_model()

        # if self.subsample is not None:
        #     n_data_subsample = min(self.subsample, len(train_indices))
        #     indices = np.array(([True] * n_data_subsample) + \
        #                        ([False] * (len(train_indices) - n_data_subsample)),
        #                        dtype=np.bool)
        #     rs = np.random.RandomState(self.seed)
        #     rs.shuffle(indices)
        #     train_indices = train_indices[indices]


        self.indices[fold] = ((train_indices, test_indices))

        if iterative and self.model.estimator_supports_iterative_fit():
            Xt, fit_params = self.model.pre_transform(self.X_train[train_indices],
                                                      self.Y_train[train_indices])

            n_iter = 1
            while not self.model.configuration_fully_fitted():
                self.model.iterative_fit(Xt, self.Y_train[train_indices],
                                         n_iter=n_iter, **fit_params)
                Y_optimization_pred, Y_valid_pred, Y_test_pred = self._predict(
                    model, train_indices=train_indices, test_indices=test_indices)
                loss = self._loss(self.Y_train[test_indices], Y_optimization_pred)
                #self.file_output(loss, Y_optimization_pred,
                #                 Y_valid_pred, Y_test_pred)
                self.finish_up(loss, )
                n_iter *= 2
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
def eval_holdout(queue, config, data, backend, seed, num_run,
                 subsample, with_predictions, all_scoring_functions,
                 output_y_test, include, exclude, disable_file_output,
                 iterative=False):
    global evaluator
    evaluator = TrainEvaluator(data, backend, config,
                               seed=seed,
                               num_run=num_run,
                               subsample=subsample,
                               with_predictions=with_predictions,
                               all_scoring_functions=all_scoring_functions,
                               output_y_test=output_y_test,
                               include=include,
                               exclude=exclude,
                               disable_file_output=disable_file_output)

    def signal_handler(signum, frame):
        print('Received signal %s. Aborting Training!' % str(signum))
        global evaluator
        duration, result, seed, run_info = evaluator.finish_up()
        queue.put((duration, result, seed, run_info, StatusType.SUCCESS))

    def empty_signal_handler(signum, frame):
        pass

    if iterative:
        signal.signal(signal.SIGALRM, signal_handler)
        evaluator.iterative_fit()
        signal.signal(signal.SIGALRM, empty_signal_handler)
        duration, result, seed, run_info = evaluator.finish_up()
    else:
        loss, opt_pred, valid_pred, test_pred = evaluator.fit_predict_and_loss()
        duration, result, seed, run_info = evaluator.finish_up(
            loss, opt_pred, valid_pred, test_pred)

    status = StatusType.SUCCESS
    queue.put((duration, result, seed, run_info, status))
