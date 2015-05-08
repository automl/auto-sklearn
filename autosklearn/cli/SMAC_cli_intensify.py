import os
import re
import sys

from autosklearn.cli import base_interface


if __name__ == "__main__":
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
        if p_name[0].startswith("-"):
            p_name = p_name[1:]
        params[p_name] = sys.argv[i + 1].strip()

    if instance_name == 'test':
        mode = 'test'
        mode_args = None
    else:
        mode = 'partial_cv'
        match = re.match(r"([0-9]+)/([0-9]+)")
        if match:
            fold = int(match.group(1))
            folds = int(match.groups(2))
        else:
            print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % (
                "ABORT", 0, 1.0, seed, "Could not parse instance name!")
            sys.exit(1)
        mode_args = {'fold': fold, 'folds': folds}

    dataset = os.path.basename(dataset_info)
    data_dir = os.path.dirname(dataset_info)

    base_interface.main(dataset, data_dir, mode, seed, params, mode_args=None)

    sys.exit(0)
