# -*- encoding: utf-8 -*-
import sys

from autosklearn.cli import base_interface


def main(output_dir=None):
    instance_name = sys.argv[1]
    instance_specific_information = sys.argv[2]
    cutoff_time = float(sys.argv[3])
    cutoff_length = int(float(sys.argv[4]))
    seed = int(float(sys.argv[5]))

    if seed < 0:
        seed = 1

    params = dict()
    for i in range(6, len(sys.argv), 2):
        p_name = str(sys.argv[i])
        if p_name[0].startswith('-'):
            p_name = p_name[1:]
        params[p_name] = sys.argv[i + 1].strip()

    if ":" in instance_name:
        instance_name = instance_name.split(":")
        mode = instance_name[0]
        mode_args = instance_name[1]
    else:
        mode = instance_name

    if mode in ('holdout', 'holdout-iterative-fit', 'test'):
        mode_args = None
    elif mode == 'nested-cv':
        mode_args = mode_args.split("/")
        inner_folds = int(mode_args[0])
        outer_folds = int(mode_args[1])
        mode_args = {'inner_folds': inner_folds, 'outer_folds': outer_folds}
    elif mode == 'partial-cv':
        mode_args = mode_args.split("/")
        fold = int(mode_args[0])
        folds = int(mode_args[1])
        mode_args = {'fold': fold, 'folds': folds}
    elif mode == 'cv':
        mode_args = {'folds': int(mode_args)}
    else:
        raise ValueError(mode)

    base_interface.main(instance_specific_information, mode,
                        seed, params, mode_args=mode_args, output_dir=output_dir)


if __name__ == '__main__':
    main()