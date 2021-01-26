import multiprocessing
import sys


def preload_modules(context: multiprocessing.context.BaseContext) -> None:
    all_loaded_modules = sys.modules.keys()
    preload = [
        loaded_module for loaded_module in all_loaded_modules
        if loaded_module.split('.')[0] in (
            'smac',
            'autosklearn',
            'numpy',
            'scipy',
            'pandas',
            'pynisher',
            'sklearn',
            'ConfigSpace',
        ) and 'logging' not in loaded_module
    ]
    context.set_forkserver_preload(preload)
