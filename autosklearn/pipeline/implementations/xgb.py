import warnings

import numpy as np

from xgboost import XGBModel, callback, rabit
from xgboost.core import DMatrix, XGBoostError, Booster, STRING_TYPES, \
    CallbackEnv, EarlyStopException
from xgboost.compat import (
    XGBClassifierBase,
    XGBRegressorBase,
    XGBLabelEncoder,
)
from xgboost.sklearn import _objective_decorator


class CustomXGBModel(XGBModel):

    def fit(self, X, y, sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True, xgb_model=None):
        # pylint: disable=missing-docstring,invalid-name,attribute-defined-outside-init
        """
        Fit the gradient boosting model
        Parameters
        ----------
        X : array_like
            Feature matrix
        y : array_like
            Labels
        sample_weight : array_like
            instance weights
        eval_set : list, optional
            A list of (X, y) tuple pairs to use as a validation set for
            early-stopping
        eval_metric : str, callable, optional
            If a str, should be a built-in evaluation metric to use. See
            doc/parameter.md. If callable, a custom evaluation metric. The call
            signature is func(y_predicted, y_true) where y_true will be a
            DMatrix object such that you may need to call the get_label
            method. It must return a str, value pair where the str is a name
            for the evaluation and value is the value of the evaluation
            function. This objective is always minimized.
        early_stopping_rounds : int
            Activates early stopping. Validation error needs to decrease at
            least every <early_stopping_rounds> round(s) to continue training.
            Requires at least one item in evals.  If there's more than one,
            will use the last. Returns the model from the last iteration
            (not the best one). If early stopping occurs, the model will
            have three additional fields: bst.best_score, bst.best_iteration
            and bst.best_ntree_limit.
            (Use bst.best_ntree_limit to get the correct value if num_parallel_tree
            and/or num_class appears in the parameters)
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation
            metric measured on the validation set to stderr.
        xgb_model : str
            file name of stored xgb model or 'Booster' instance Xgb model to be
            loaded before training (allows training continuation).
        """
        if sample_weight is not None:
            trainDmatrix = DMatrix(X, label=y, weight=sample_weight,
                                   missing=self.missing, nthread=self.n_jobs)
        else:
            trainDmatrix = DMatrix(X, label=y, missing=self.missing,
                                   nthread=self.n_jobs)

        evals_result = {}
        if eval_set is not None:
            evals = list(DMatrix(x[0], label=x[1], missing=self.missing,
                                 nthread=self.n_jobs) for x in eval_set)
            evals = list(zip(evals, ["validation_{}".format(i) for i in
                                     range(len(evals))]))
        else:
            evals = ()

        params = self.get_xgb_params()

        if callable(self.objective):
            obj = _objective_decorator(self.objective)
            params["objective"] = "reg:linear"
        else:
            obj = None

        feval = eval_metric if callable(eval_metric) else None
        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            else:
                params.update({'eval_metric': eval_metric})

        self._Booster = train(params, trainDmatrix,
                              self.n_estimators, evals=evals,
                              early_stopping_rounds=early_stopping_rounds,
                              evals_result=evals_result, obj=obj, feval=feval,
                              verbose_eval=verbose, xgb_model=xgb_model)

        if evals_result:
            for val in evals_result.items():
                evals_result_key = list(val[1].keys())[0]
                evals_result[val[0]][evals_result_key] = val[1][
                    evals_result_key]
            self.evals_result_ = evals_result

        if early_stopping_rounds is not None:
            self.best_score = self._Booster.best_score
            self.best_iteration = self._Booster.best_iteration
            self.best_ntree_limit = self._Booster.best_ntree_limit
        return self


class CustomXGBClassifier(CustomXGBModel, XGBClassifierBase):
    # pylint: disable=missing-docstring,too-many-arguments,invalid-name
    __doc__ = """Implementation of the scikit-learn API for XGBoost classification.

    """ + '\n'.join(XGBModel.__doc__.split('\n')[2:])

    def __init__(self, max_depth=3, learning_rate=0.1,
                 n_estimators=100, silent=True,
                 objective="binary:logistic", booster='gbtree',
                 n_jobs=1, nthread=None, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1,
                 colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, random_state=0, seed=None, missing=None,
                 **kwargs):
        super(CustomXGBClassifier, self).__init__(
            max_depth, learning_rate,
            n_estimators, silent, objective,
            booster,
            n_jobs, nthread, gamma,
            min_child_weight,
            max_delta_step, subsample,
            colsample_bytree, colsample_bylevel,
            reg_alpha, reg_lambda,
            scale_pos_weight, base_score,
            random_state, seed, missing,
            **kwargs
        )

    def fit(self, X, y, sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True, xgb_model=None):
        # pylint: disable = attribute-defined-outside-init,arguments-differ
        """
        Fit gradient boosting classifier

        Parameters
        ----------
        X : array_like
            Feature matrix
        y : array_like
            Labels
        sample_weight : array_like
            Weight for each instance
        eval_set : list, optional
            A list of (X, y) pairs to use as a validation set for
            early-stopping
        eval_metric : str, callable, optional
            If a str, should be a built-in evaluation metric to use. See
            doc/parameter.md. If callable, a custom evaluation metric. The call
            signature is func(y_predicted, y_true) where y_true will be a
            DMatrix object such that you may need to call the get_label
            method. It must return a str, value pair where the str is a name
            for the evaluation and value is the value of the evaluation
            function. This objective is always minimized.
        early_stopping_rounds : int, optional
            Activates early stopping. Validation error needs to decrease at
            least every <early_stopping_rounds> round(s) to continue training.
            Requires at least one item in evals.  If there's more than one,
            will use the last. Returns the model from the last iteration
            (not the best one). If early stopping occurs, the model will
            have three additional fields: bst.best_score, bst.best_iteration
            and bst.best_ntree_limit.
            (Use bst.best_ntree_limit to get the correct value if num_parallel_tree
            and/or num_class appears in the parameters)
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation
            metric measured on the validation set to stderr.
        xgb_model : str
            file name of stored xgb model or 'Booster' instance Xgb model to be
            loaded before training (allows training continuation).
        """
        evals_result = {}
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        xgb_options = self.get_xgb_params()

        if callable(self.objective):
            obj = _objective_decorator(self.objective)
            # Use default value. Is it really not used ?
            xgb_options["objective"] = "binary:logistic"
        else:
            obj = None

        if self.n_classes_ > 2:
            # Switch to using a multiclass objective in the underlying XGB instance
            xgb_options["objective"] = "multi:softprob"
            xgb_options['num_class'] = self.n_classes_

        feval = eval_metric if callable(eval_metric) else None
        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            else:
                xgb_options.update({"eval_metric": eval_metric})

        self._le = XGBLabelEncoder().fit(y)
        training_labels = self._le.transform(y)

        if eval_set is not None:
            # TODO: use sample_weight if given?
            evals = list(
                DMatrix(x[0], label=self._le.transform(x[1]),
                        missing=self.missing, nthread=self.n_jobs)
                for x in eval_set
            )
            nevals = len(evals)
            eval_names = ["validation_{}".format(i) for i in range(nevals)]
            evals = list(zip(evals, eval_names))
        else:
            evals = ()

        self._features_count = X.shape[1]

        if sample_weight is not None:
            train_dmatrix = DMatrix(X, label=training_labels,
                                    weight=sample_weight,
                                    missing=self.missing, nthread=self.n_jobs)
        else:
            train_dmatrix = DMatrix(X, label=training_labels,
                                    missing=self.missing, nthread=self.n_jobs)

        self._Booster = train(xgb_options, train_dmatrix, self.n_estimators,
                              evals=evals,
                              early_stopping_rounds=early_stopping_rounds,
                              evals_result=evals_result, obj=obj, feval=feval,
                              # Only the last kwarg in of this call was
                              # changed in this file!!!
                              verbose_eval=verbose, xgb_model=xgb_model)

        self.objective = xgb_options["objective"]
        if evals_result:
            for val in evals_result.items():
                evals_result_key = list(val[1].keys())[0]
                evals_result[val[0]][evals_result_key] = val[1][
                    evals_result_key]
            self.evals_result_ = evals_result

        if early_stopping_rounds is not None:
            self.best_score = self._Booster.best_score
            self.best_iteration = self._Booster.best_iteration
            self.best_ntree_limit = self._Booster.best_ntree_limit

        return self

    def predict(self, data, output_margin=False, ntree_limit=0):
        test_dmatrix = DMatrix(data, missing=self.missing, nthread=self.n_jobs)
        class_probs = self.get_booster().predict(test_dmatrix,
                                                 output_margin=output_margin,
                                                 ntree_limit=ntree_limit)
        if len(class_probs.shape) > 1:
            column_indexes = np.argmax(class_probs, axis=1)
        else:
            column_indexes = np.repeat(0, class_probs.shape[0])
            column_indexes[class_probs > 0.5] = 1
        return self._le.inverse_transform(column_indexes)

    def predict_proba(self, data, output_margin=False, ntree_limit=0):
        test_dmatrix = DMatrix(data, missing=self.missing, nthread=self.n_jobs)
        class_probs = self.get_booster().predict(test_dmatrix,
                                                 output_margin=output_margin,
                                                 ntree_limit=ntree_limit)
        if self.objective == "multi:softprob":
            return class_probs
        else:
            classone_probs = class_probs
            classzero_probs = 1.0 - classone_probs
            return np.vstack((classzero_probs, classone_probs)).transpose()

    def evals_result(self):
        """Return the evaluation results.

        If eval_set is passed to the `fit` function, you can call evals_result() to
        get evaluation results for all passed eval_sets. When eval_metric is also
        passed to the `fit` function, the evals_result will contain the eval_metrics
        passed to the `fit` function

        Returns
        -------
        evals_result : dictionary

        Example
        -------
        param_dist = {'objective':'binary:logistic', 'n_estimators':2}

        clf = xgb.XGBClassifier(**param_dist)

        clf.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                eval_metric='logloss',
                verbose=True)

        evals_result = clf.evals_result()

        The variable evals_result will contain:
        {'validation_0': {'logloss': ['0.604835', '0.531479']},
         'validation_1': {'logloss': ['0.41965', '0.17686']}}
        """
        if self.evals_result_:
            evals_result = self.evals_result_
        else:
            raise XGBoostError('No results.')

        return evals_result


class CustomXGBRegressor(CustomXGBModel, XGBRegressorBase):
    # pylint: disable=missing-docstring
    __doc__ = """Implementation of the scikit-learn API for XGBoost regression.
    """ + '\n'.join(XGBModel.__doc__.split('\n')[2:])


def _train_internal(params, dtrain,
                    num_boost_round=10, evals=(),
                    obj=None, feval=None,
                    xgb_model=None, callbacks=None):
    """internal training function"""
    callbacks = [] if callbacks is None else callbacks
    evals = list(evals)
    if isinstance(params, dict) \
            and 'eval_metric' in params \
            and isinstance(params['eval_metric'], list):
        params = dict((k, v) for k, v in params.items())
        eval_metrics = params['eval_metric']
        params.pop("eval_metric", None)
        params = list(params.items())
        for eval_metric in eval_metrics:
            params += [('eval_metric', eval_metric)]

    bst = Booster(params, [dtrain] + [d[0] for d in evals])
    nboost = 0
    num_parallel_tree = 1

    if xgb_model is not None:
        if not isinstance(xgb_model, STRING_TYPES):
            xgb_model = xgb_model.save_raw()
        bst = Booster(params, [dtrain] + [d[0] for d in evals],
                      model_file=xgb_model)
        nboost = len(bst.get_dump())

    _params = dict(params) if isinstance(params, list) else params

    if 'num_parallel_tree' in _params:
        num_parallel_tree = _params['num_parallel_tree']
        nboost //= num_parallel_tree
    if 'num_class' in _params:
        nboost //= _params['num_class']

    # Distributed code: Load the checkpoint from rabit.
    version = bst.load_rabit_checkpoint()
    assert (rabit.get_world_size() != 1 or version == 0)
    rank = rabit.get_rank()
    start_iteration = int(version / 2)
    nboost += start_iteration

    callbacks_before_iter = [
        cb for cb in callbacks if cb.__dict__.get('before_iteration', False)]
    callbacks_after_iter = [
        cb for cb in callbacks if
        not cb.__dict__.get('before_iteration', False)]

    for i in range(nboost, num_boost_round):
        for cb in callbacks_before_iter:
            cb(CallbackEnv(model=bst,
                           cvfolds=None,
                           iteration=i,
                           begin_iteration=start_iteration,
                           end_iteration=num_boost_round,
                           rank=rank,
                           evaluation_result_list=None))
        # Distributed code: need to resume to this point.
        # Skip the first update if it is a recovery step.
        if version % 2 == 0:
            bst.update(dtrain, i, obj)
            bst.save_rabit_checkpoint()
            version += 1

        assert (
        rabit.get_world_size() == 1 or version == rabit.version_number())

        nboost += 1
        evaluation_result_list = []
        # check evaluation result.
        if len(evals) != 0:
            bst_eval_set = bst.eval_set(evals, i, feval)
            if isinstance(bst_eval_set, STRING_TYPES):
                msg = bst_eval_set
            else:
                msg = bst_eval_set.decode()
            res = [x.split(':') for x in msg.split()]
            evaluation_result_list = [(k, float(v)) for k, v in res[1:]]
        try:
            for cb in callbacks_after_iter:
                cb(CallbackEnv(model=bst,
                               cvfolds=None,
                               iteration=i,
                               begin_iteration=start_iteration,
                               end_iteration=num_boost_round,
                               rank=rank,
                               evaluation_result_list=evaluation_result_list))
        except EarlyStopException:
            break
        # do checkpoint after evaluation, in case evaluation also updates booster.
        bst.save_rabit_checkpoint()
        version += 1

    if bst.attr('best_score') is not None:
        bst.best_score = float(bst.attr('best_score'))
        bst.best_iteration = int(bst.attr('best_iteration'))
    else:
        bst.best_iteration = nboost - 1
    bst.best_ntree_limit = (bst.best_iteration + 1) * num_parallel_tree
    return bst


def train(params, dtrain, num_boost_round=10, evals=(), obj=None, feval=None,
          maximize=False, early_stopping_rounds=None, evals_result=None,
          verbose_eval=True, xgb_model=None, callbacks=None,
          learning_rates=None):
    # pylint: disable=too-many-statements,too-many-branches, attribute-defined-outside-init
    """Train a booster with given parameters.

    Parameters
    ----------
    params : dict
        Booster params.
    dtrain : DMatrix
        Data to be trained.
    num_boost_round: int
        Number of boosting iterations.
    evals: list of pairs (DMatrix, string)
        List of items to be evaluated during training, this allows user to watch
        performance on the validation set.
    obj : function
        Customized objective function.
    feval : function
        Customized evaluation function.
    maximize : bool
        Whether to maximize feval.
    early_stopping_rounds: int
        Activates early stopping. Validation error needs to decrease at least
        every <early_stopping_rounds> round(s) to continue training.
        Requires at least one item in evals.
        If there's more than one, will use the last.
        Returns the model from the last iteration (not the best one).
        If early stopping occurs, the model will have three additional fields:
        bst.best_score, bst.best_iteration and bst.best_ntree_limit.
        (Use bst.best_ntree_limit to get the correct value if num_parallel_tree
        and/or num_class appears in the parameters)
    evals_result: dict
        This dictionary stores the evaluation results of all the items in watchlist.
        Example: with a watchlist containing [(dtest,'eval'), (dtrain,'train')] and
        a parameter containing ('eval_metric': 'logloss')
        Returns: {'train': {'logloss': ['0.48253', '0.35953']},
                  'eval': {'logloss': ['0.480385', '0.357756']}}
    verbose_eval : bool or int
        Requires at least one item in evals.
        If `verbose_eval` is True then the evaluation metric on the validation set is
        printed at each boosting stage.
        If `verbose_eval` is an integer then the evaluation metric on the validation set
        is printed at every given `verbose_eval` boosting stage. The last boosting stage
        / the boosting stage found by using `early_stopping_rounds` is also printed.
        Example: with verbose_eval=4 and at least one item in evals, an evaluation metric
        is printed every 4 boosting stages, instead of every boosting stage.
    learning_rates: list or function (deprecated - use callback API instead)
        List of learning rate for each boosting round
        or a customized function that calculates eta in terms of
        current number of round and the total number of boosting round (e.g. yields
        learning rate decay)
    xgb_model : file name of stored xgb model or 'Booster' instance
        Xgb model to be loaded before training (allows training continuation).
    callbacks : list of callback functions
        List of callback functions that are applied at end of each iteration.
        It is possible to use predefined callbacks by using xgb.callback module.
        Example: [xgb.callback.reset_learning_rate(custom_rates)]

    Returns
    -------
    booster : a trained booster model
    """
    callbacks = [] if callbacks is None else callbacks

    # Most of legacy advanced options becomes callbacks
    if isinstance(verbose_eval, bool) and verbose_eval:
        callbacks.append(callback.print_evaluation())
    else:
        if isinstance(verbose_eval, int):
            callbacks.append(callback.print_evaluation(verbose_eval))

    if early_stopping_rounds is not None:
        callbacks.append(callback.early_stop(early_stopping_rounds,
                                             maximize=maximize,
                                             verbose=bool(verbose_eval)))
    if evals_result is not None:
        callbacks.append(callback.record_evaluation(evals_result))

    if learning_rates is not None:
        warnings.warn(
            "learning_rates parameter is deprecated - use callback API instead",
            DeprecationWarning)
        callbacks.append(callback.reset_learning_rate(learning_rates))

    return _train_internal(params, dtrain,
                           num_boost_round=num_boost_round,
                           evals=evals,
                           obj=obj, feval=feval,
                           xgb_model=xgb_model, callbacks=callbacks)