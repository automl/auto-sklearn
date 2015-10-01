# -*- encoding: utf-8 -*-
from __future__ import print_function

import os
import signal
import time

import lockfile
import six.moves.cPickle as pickle

from HPOlibConfigSpace import configuration_space

from autosklearn.data.competition_data_manager import CompetitionDataManager
from autosklearn.evaluation import CVEvaluator, HoldoutEvaluator, \
    NestedCVEvaluator, TestEvaluator, get_new_run_num
from autosklearn.util.paramsklearn import get_configuration_space
from autosklearn.util import Backend



def store_and_or_load_data(dataset_info, outputdir):
    backend = Backend(None, outputdir)

    try:
        D = backend.load_datamanager()
    except IOError:
        D = None

    # Datamanager probably doesn't exist
    if D is None:
        D = CompetitionDataManager(dataset_info, encode_labels=True)
        backend.save_datamanager(D)

    return D

# signal handler seem to work only if they are globally defined
# to give it access to the evaluator class, the evaluator name has to
# be a global name. It's not the cleanest solution, but works for now.
evaluator = None


def signal_handler(signum, frame):
    print('Aborting Training!')
    global evaluator
    evaluator.finish_up()
    exit(0)


def empty_signal_handler(signum, frame):
    print('Received Signal %s, but alread finishing up!' % str(signum))


signal.signal(15, signal_handler)


def _get_base_dict():
    return {
        'with_predictions': True,
        'all_scoring_functions': True,
        'output_y_test': True,
    }


def make_mode_holdout(data, seed, configuration, num_run):
    evaluator = HoldoutEvaluator(data, configuration,
                                 seed=seed,
                                 num_run=num_run,
                                 **_get_base_dict())
    evaluator.fit()
    signal.signal(15, empty_signal_handler)
    evaluator.finish_up()

    backend = Backend(None, os.getcwd())
    if os.path.exists(backend.get_model_dir()):
        backend.save_model(evaluator.model, num_run)


def make_mode_test(data, seed, configuration, metric):
    evaluator = TestEvaluator(data,
                              configuration,
                              all_scoring_functions=True,
                              seed=seed)
    evaluator.fit()
    scores = evaluator.predict()
    duration = time.time() - evaluator.starttime

    score = scores[metric]
    additional_run_info = ';'.join(['%s: %s' % (m_, value)
                                    for m_, value in scores.items()])
    additional_run_info += ';' + 'duration: ' + str(duration)

    print('Result for ParamILS: %s, %f, 1, %f, %d, %s' %
          ('SAT', abs(duration), score, evaluator.seed,
           additional_run_info))


def make_mode_cv(data, seed, configuration, num_run, folds):
    evaluator = CVEvaluator(data, configuration,
                            cv_folds=folds,
                            seed=seed,
                            num_run=num_run,
                            **_get_base_dict())
    evaluator.fit()
    signal.signal(15, empty_signal_handler)
    evaluator.finish_up()


def make_mode_partial_cv(data, seed, configuration, num_run, metric, fold,
                         folds):
    evaluator = CVEvaluator(data, configuration,
                            all_scoring_functions=True,
                            cv_folds=folds,
                            seed=seed,
                            num_run=num_run)
    evaluator.partial_fit(fold)
    scores = evaluator.predict()
    duration = time.time() - evaluator.starttime

    score = scores[metric]
    additional_run_info = ';'.join(['%s: %s' % (m_, value)
                                    for m_, value in scores.items()])
    additional_run_info += ';' + 'duration: ' + str(duration)

    print('Result for ParamILS: %s, %f, 1, %f, %d, %s' %
          ('SAT', abs(duration), score, evaluator.seed,
           additional_run_info))


def make_mode_nested_cv(data, seed, configuration, num_run, inner_folds,
                        outer_folds):
    evaluator = NestedCVEvaluator(data, configuration,
                                  inner_cv_folds=inner_folds,
                                  outer_cv_folds=outer_folds,
                                  seed=seed,
                                  num_run=num_run,
                                  **_get_base_dict())
    evaluator.fit()
    signal.signal(15, empty_signal_handler)
    evaluator.finish_up()


def main(dataset_info, mode, seed, params, mode_args=None):
    """This command line interface has three different operation modes:

    * CV: useful for the Tweakathon
    * 1/3 test split: useful to evaluate a configuration
    * cv on 2/3 train split: useful to optimize hyperparameters in a training
      mode before testing a configuration on the 1/3 test split.

    It must by no means be used for the Auto part of the competition!
    """
    if mode_args is None:
        mode_args = {}

    num_run = None
    if mode != 'test':
        num_run = get_new_run_num()

    for key in params:
        try:
            params[key] = int(params[key])
        except Exception:
            try:
                params[key] = float(params[key])
            except Exception:
                pass

    if seed is not None:
        seed = int(float(seed))
    else:
        seed = 1

    output_dir = os.getcwd()

    D = store_and_or_load_data(dataset_info=dataset_info, outputdir=output_dir)

    cs = get_configuration_space(D.info)
    configuration = configuration_space.Configuration(cs, params)
    metric = D.info['metric']

    global evaluator

    if mode == 'holdout':
        make_mode_holdout(D, seed, configuration, num_run)
    elif mode == 'test':
        make_mode_test(D, seed, configuration, metric)
    elif mode == 'cv':
        make_mode_cv(D, seed, configuration, num_run, mode_args['folds'])
    elif mode == 'partial-cv':
        make_mode_partial_cv(D, seed, configuration, num_run,
                             metric, mode_args['fold'], mode_args['folds'])
    elif mode == 'nested-cv':
        make_mode_nested_cv(D, seed, configuration, num_run,
                            mode_args['inner_folds'], mode_args['outer_folds'])
    else:
        raise ValueError('Must choose a legal mode.')
