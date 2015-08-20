# -*- encoding: utf-8 -*-

import os
import sys


def check_pid(pid):
    """Check For the existence of a unix pid."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def get_info_from_file(datadir, dataset):
    """
    Get all information {attribute = value} pairs from the public.info file
    :param datadir:
    :param dataset:
    :return:
    """
    dataset_path = os.path.join(datadir, dataset, dataset + '_public.info')

    if not os.path.exists(dataset_path):
        sys.stderr.write('Could not find info: %s\n' % dataset_path)
        return {'time_budget': 600}

    info = dict()

    with open(dataset_path, 'r') as info_file:
        lines = info_file.readlines()
        features_list = list(map(lambda x: tuple(x.strip("\'").split(' = ')),
                                 lines))

        for (key, value) in features_list:
            info[key] = value.rstrip().strip("'").strip(' ')
            # if we have a number, we want it to be an integer
            if info[key].isdigit():
                info[key] = int(info[key])
    return info
