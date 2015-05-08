import time
import re

try:
    from HPOlib.benchmark_util import parse_cli
except:
    from HPOlib.benchmarks.benchmark_util import parse_cli

from autosklearn.cli import base_interface


if __name__ == "__main__":
    starttime = time.time()
    args, params = parse_cli()

    dataset = args['dataset']
    data_dir = args['data_dir']
    mode = args['mode']
    seed = args.get('seed')
    fold = int(args['fold'])
    folds = int(args['folds'])

    if mode.endswith("cv"):
        if folds == 1:
            cv_match = re.match(r"([0-9]*)cv", mode)
            real_folds = cv_match.group(1)
            real_folds = 10 if not real_folds else int(real_folds)
            mode = 'cv'
            mode_args = {'folds': real_folds}
        else:
            mode = 'partial_cv'
            mode_args = {'fold': fold, 'folds': folds}
    elif mode == 'holdout':
        mode_args = None
    elif mode == 'test':
        mode_args = None
    else:
        raise ValueError(mode)

    base_interface.main(dataset, data_dir, mode, seed, params, mode_args=mode_args)
