# -*- encoding: utf-8 -*-
import sys

from autosklearn.cli import base_interface

if __name__ == '__main__':
    dataset_info = sys.argv[1]
    instance_name = sys.argv[2]
    instance_specific_information = sys.argv[3]
    cutoff_time = float(sys.argv[4])
    cutoff_length = int(float(sys.argv[5]))
    seed = int(float(sys.argv[6]))

    if seed < 0:
        seed = 1

    params = dict()
    for i in range(7, len(sys.argv), 2):
        p_name = str(sys.argv[i])
        if p_name[0].startswith('-'):
            p_name = p_name[1:]
        params[p_name] = sys.argv[i + 1].strip()

    mode_args = None
    mode = 'test' if instance_name == 'test' else 'holdout'
    base_interface.main(dataset_info, mode, seed, params, mode_args=None)

    sys.exit(0)
