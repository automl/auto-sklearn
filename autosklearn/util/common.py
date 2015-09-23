# -*- encoding: utf-8 -*-


import os
import sys


__all__ = [
    'check_pid',
    'get_info_from_file',
    'set_auto_seed',
    'get_auto_seed',
    'del_auto_seed',
]


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


def _set_get_del_env_key(key):
    env_key = key

    def set_value(value):
        env_value = os.environ.get(env_key)
        if env_value is not None:
            raise ValueError('It seems you have already started an instance '
                             'in this thread.')
        else:
            os.environ[env_key] = str(value)

    def get_value():
        value = os.environ.get(env_key, None)
        assert value is not None, "Not found %s in env variables" % env_key
        return int(value)

    def del_value():
        if env_key in os.environ:
            del os.environ[env_key]

    return set_value, get_value, del_value


set_auto_seed, get_auto_seed, del_auto_seed = _set_get_del_env_key(
    "AUTOSKLEARN_SEED")
