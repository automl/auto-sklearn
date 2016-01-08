# -*- encoding: utf-8 -*-
import re
import sys
import time

from autosklearn.cli import base_interface


def parse_cli():
    """
    Provide a generic command line interface for benchmarks. It will just parse
    the command line according to simple rules and return two dictionaries, one
    containing all arguments for the benchmark algorithm like dataset,
    crossvalidation metadata etc. and the containing all learning algorithm
    hyperparameters.
    Parsing rules:
    - Arguments with two minus signs are treated as benchmark arguments, Xalues
     are not allowed to start with a minus. The last argument must --params,
     starting the hyperparameter arguments.
    - All arguments after --params are treated as hyperparameters to the
     learning algorithm. Every parameter name must start with one minus and must
     have exactly one value which has to be given in single quotes.
    - Arguments with no value before --params are treated as boolean arguments
    Example:
    python neural_network.py --folds 10 --fold 1 --dataset convex  --params
        -depth '3' -n_hid_0 '1024' -n_hid_1 '1024' -n_hid_2 '1024' -lr '0.01'
    """
    args = {}
    arg_name = None
    arg_values = None
    parameters = {}

    cli_args = sys.argv
    found_params = False
    skip = True
    iterator = enumerate(cli_args)

    for idx, arg in iterator:
        if skip:
            skip = False
            continue
        else:
            skip = True

        if arg == "--params":
            if arg_name:
                args[arg_name] = " ".join(arg_values)
            found_params = True
            skip = False

        elif arg[0:2] == "--" and not found_params:
            if arg_name:
                args[arg_name] = " ".join(arg_values)
            arg_name = arg[2:]
            arg_values = []
            skip = False

        elif arg[0:2] == "--" and found_params:
            raise ValueError("You are trying to specify an argument after the "
                             "--params argument. Please change the order.")

        elif arg[0] == "-" and arg[0:2] != "--" and found_params:
            parameters[cli_args[idx][1:]] = cli_args[idx + 1]

        elif arg[0] == "-" and arg[0:2] != "--" and not found_params:
            raise ValueError("You either try to use arguments with only one lea"
                             "ding minus or try to specify a hyperparameter bef"
                             "ore the --params argument. %s" %
                             " ".join(cli_args))
        elif arg[0:2] != "--" and not found_params:
            arg_values.append(arg)
            skip = False

        elif not found_params:
            raise ValueError("Illegal command line string, expected an argument"
                             " starting with -- but found %s" % (arg,))

        else:
            raise ValueError("Illegal command line string, expected a hyperpara"
                             "meter starting with - but found %s" % (arg,))

    return args, parameters


def parse_args(dataset, mode, seed, params, fold, folds, output_dir=None):
    if seed is None:
        seed = 1

    if 'nested-cv' in mode:
        # Specifiy like this 5/5-nested-cv
        cv_match = re.match(r"([0-9]+)/([0-9]+)-nested-cv", mode)
        outer_folds = int(cv_match.group(1))
        inner_folds = int(cv_match.group(2))
        mode = 'nested-cv'
        mode_args = {'inner_folds': inner_folds, 'outer_folds': outer_folds}
    elif mode.endswith('cv'):
        if folds == 1:
            cv_match = re.match(r"([0-9]*)cv", mode)
            real_folds = cv_match.group(1)
            real_folds = 10 if not real_folds else int(real_folds)
            mode = 'cv'
            mode_args = {'folds': real_folds}
        else:
            mode = 'partial-cv'
            mode_args = {'fold': fold, 'folds': folds}
    elif mode in ('holdout', 'holdout-iterative-fit', 'test'):
        mode_args = None
    else:
        raise ValueError(mode)
    base_interface.main(dataset, mode, seed, params, mode_args=mode_args,
                        output_dir=output_dir)


def main(output_dir=None):
    args, params = parse_cli()
    assert 'dataset' in args
    assert 'mode' in args
    assert 'seed' in args
    assert 'fold' in args and type(int(args['fold'])) == int
    assert 'folds' in args and type(int(args['folds'])) == int

    parse_args(args['dataset'],
               args['mode'],
               args.get('seed'),
               params,
               int(args['fold']),
               int(args['folds']),
               output_dir=output_dir
        )


if __name__ == '__main__':
    main()
