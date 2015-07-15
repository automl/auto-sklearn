import argparse
import logging
from itertools import product
import functools

import pyMetaLearn.optimizers.optimizer_base as optimizer_base


logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Gridsearch")
logger.setLevel(logging.INFO)


def parse_parameters(args=None):
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-a", "--algorithm")
    group.add_argument("--cli_target")
    parser.add_argument("-p", "--params", required=True)
    args = parser.parse_args(args=args)
    return args


def perform_gridsearch(fn, grid):
    retvals = []
    for idx, parameters in enumerate(grid):
        logger.info("%d/%d, parameters: %s\n" % (idx+1, len(grid),
                                                 str(parameters)))
        retvals.append(fn(parameters))
        logger.info("Response: " + str(retvals[-1]))
    return min(retvals)


def main(args=None):
    args = parse_parameters()
    fh = open(args.params)
    param_string = fh.read()
    fh.close()
    hyperparameters = optimizer_base.parse_hyperparameter_string(param_string)
    grid = optimizer_base.build_grid(hyperparameters)
    if args.algorithm:
        raise NotImplementedError()
    elif args.cli_target:
        cli_function = optimizer_base.command_line_function()
        fn = functools.partial(cli_function, args.cli_target)
    #print perform_gridsearch(fn, grid)
    perform_gridsearch(fn, grid)


if __name__ == "__main__":
    main()
