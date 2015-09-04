# -*- encoding: utf-8 -*-

import os

__all__ = [
    'set_auto_seed',
    'get_auto_seed',
    'del_auto_seed',
]

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
