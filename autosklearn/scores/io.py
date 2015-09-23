# -*- encoding: utf-8 -*-
from __future__ import print_function

import os
import platform
from glob import glob
from os import getcwd as pwd
from sys import stderr, version

from pip import get_installed_distributions as lib
import psutil
import yaml
from autosklearn.scores.classification_metrics import auc_metric, acc_metric
from autosklearn.scores.regression_metrics import r2_metric, a_metric
from autosklearn.scores.specialized_scores import npac_multiclass_score, \
    npac_binary_score, f1_multiclass_score, f1_binary_score, \
    nbac_multiclass_score, nbac_binary_score
from autosklearn.scores.util import sanitize_array, normalize_array

swrite = stderr.write


def ls(filename):
    return sorted(glob(filename))


def write_list(lst):
    for item in lst:
        swrite(item + '\n')


def mkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def get_info(filename):
    """
    Get all information {attribute = value} pairs from the public.info file
    :param filename:
    :return:
    """
    info = {}
    with open(filename, 'r') as info_file:
        lines = info_file.readlines()
        features_list = list(map(lambda x: tuple(x.strip("\'").split(' = ')),
                                 lines))
        for (key, value) in features_list:
            info[key] = value.rstrip().strip("'").strip(' ')
            # if we have a number, we want it to be an integer
            if info[key].isdigit():
                info[key] = int(info[key])
    return info


def show_io(input_dir, output_dir):
    """
    show directory structure and inputs and autputs to scoring program.
    :param input_dir:
    :param output_dir:
    :return:
    """
    swrite('\n=== DIRECTORIES ===\n\n')
    # Show this directory
    swrite('-- Current directory ' + pwd() + ':\n')
    write_list(ls('.'))
    write_list(ls('./*'))
    write_list(ls('./*/*'))
    swrite('\n')

    # List input and output directories
    swrite('-- Input directory ' + input_dir + ':\n')
    write_list(ls(input_dir))
    write_list(ls(input_dir + '/*'))
    write_list(ls(input_dir + '/*/*'))
    write_list(ls(input_dir + '/*/*/*'))
    swrite('\n')
    swrite('-- Output directory  ' + output_dir + ':\n')
    write_list(ls(output_dir))
    write_list(ls(output_dir + '/*'))
    swrite('\n')

    # write meta data to sdterr
    swrite('\n=== METADATA ===\n\n')
    swrite('-- Current directory ' + pwd() + ':\n')
    try:
        metadata = yaml.load(open('metadata', 'r'))
        for key, value in metadata.items():
            swrite(key + ': ')
            swrite(str(value) + '\n')
    except Exception:
        swrite('none\n')
    swrite('-- Input directory ' + input_dir + ':\n')
    try:
        metadata = yaml.load(open(os.path.join(input_dir, 'metadata'), 'r'))
        for key, value in metadata.items():
            swrite(key + ': ')
            swrite(str(value) + '\n')
        swrite('\n')
    except Exception:
        swrite('none\n')


def show_version(scoring_version):
    """
    Python version and library versions.
    :param scoring_version:
    :return:
    """
    swrite('\n=== VERSIONS ===\n\n')
    # Scoring program version
    swrite('Scoring program version: ' + str(scoring_version) + '\n\n')
    # Python version
    swrite('Python version: ' + version + '\n\n')
    # Give information on the version installed
    swrite('Versions of libraries installed:\n')
    map(swrite, sorted(['%s==%s\n' % (i.key, i.version) for i in lib()]))


def show_platform():
    """
    Show information on platform.
    :return:
    """
    swrite('\n=== SYSTEM ===\n\n')
    try:
        linux_distribution = platform.linux_distribution()
    except Exception:
        linux_distribution = 'N/A'
    swrite("""
    dist: %s
    linux_distribution: %s
    system: %s
    machine: %s
    platform: %s
    uname: %s
    version: %s
    mac_ver: %s
    memory: %s
    number of CPU: %s
    """ % (str(platform.dist()), linux_distribution, platform.system(),
           platform.machine(), platform.platform(), platform.uname(),
           platform.version(), platform.mac_ver(), psutil.virtual_memory(),
           str(psutil.cpu_count())))


def compute_all_scores(solution, prediction):
    """
    Compute all the scores and return them as a dist.
    :param solution:
    :param prediction:
    :return:
    """
    missing_score = -0.999999
    scoring = {
        'ACC': acc_metric,
        'BAC (multilabel)': nbac_binary_score,
        'BAC (multiclass)': nbac_multiclass_score,
        'F1  (multilabel)': f1_binary_score,
        'F1  (multiclass)': f1_multiclass_score,
        'Regression ABS  ': a_metric,
        'Regression R2   ': r2_metric,
        'AUC (multilabel)': auc_metric,
        'PAC (multilabel)': npac_binary_score,
        'PAC (multiclass)': npac_multiclass_score
    }
    # Normalize/sanitize inputs
    [csolution, cprediction] = normalize_array(solution, prediction)
    solution = sanitize_array(solution)
    prediction = sanitize_array(prediction)
    # Compute all scores
    score_names = sorted(scoring.keys())
    scores = {}
    for key in score_names:
        scoring_func = scoring[key]
        try:
            if key == 'Regression R2   ' or key == 'Regression ABS  ':
                scores[key] = scoring_func(solution, prediction)
            else:
                scores[key] = scoring_func(csolution, cprediction)
        except Exception:
            scores[key] = missing_score
    return scores


def write_scores(fp, scores):
    """
    Write scores to file opened under file pointer fp.
    :param fp:
    :param scores:
    :return:
    """
    for key in scores.keys():
        fp.write('%s --> %s\n' % (key, scores[key]))
        print(key + ' --> ' + str(scores[key]))


def show_all_scores(solution, prediction):
    """
    Compute and display all the scores for debug purposes.
    :param solution:
    :param prediction:
    :return:
    """
    scores = compute_all_scores(solution, prediction)
    for key in scores.keys():
        print(key + ' --> ' + str(scores[key]))
