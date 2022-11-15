import multiprocessing
import sys


def preload_modules(context: multiprocessing.context.BaseContext) -> None:
    """Attempt to preload modules when using forkserver"""
    # NOTE: preloading and docstring
    #
    #   This is just a best guess at why this is used, coming from this blogpost
    #   https://bnikolic.co.uk/blog/python/parallelism/2019/11/13/python-forkserver-preload.html
    #   Ideally we should identify subprocesses that get run with this and try limit the
    #   necessity to use all of these modules
    #
    #   @eddiebergman
    all_loaded_modules = list(sys.modules.keys())
    preload = [
        loaded_module
        for loaded_module in all_loaded_modules
        if loaded_module.split(".")[0]
        in (
            "smac",
            "autosklearn",
            "numpy",
            "scipy",
            "pandas",
            "pynisher",
            "sklearn",
            "ConfigSpace",
        )
        and "logging" not in loaded_module
    ]
    context.set_forkserver_preload(preload)
