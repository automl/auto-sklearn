import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, required=True,
                        help="File with all datasets.")
    parser.add_argument("--runs-per-dataset", default=1, type=int,
                        help="Number of configuration runs per dataset.")
    parser.add_argument("--output-directory", type=str, required=True,
                        help="Configuration output directory.")
    parser.add_argument("--time-limit", type=int, required=True,
                        help="Total configuration time limit.")
    parser.add_argument("--per-run-time-limit", type=int, required=True,
                        help="Time limit for an individual run.")
    parser.add_argument("--ml-memory-limit", type=int, required=True,
                        help="Memory limit for the target algorith run.")
    args = parser.parse_args()

    datasets_dir = os.path.abspath(os.path.dirname(args.datasets))
    datasets = []
    with open(args.datasets, 'r') as fh:
        for line in fh:
            line = line.strip()
            dataset = os.path.join(datasets_dir, line)
            datasets.append(dataset)

    commands = []
    for num_run in range(args.runs_per_dataset):
        for dataset in datasets:
            if dataset[-1] == "/":
                dataset = dataset[:-1]
            dataset_name = os.path.basename(dataset)
            output_directory = os.path.join(args.output_directory, dataset_name)
            try:
                os.makedirs(output_directory)
            except:
                pass
            command = 'autosklearn --output-dir %s ' \
                      '--temporary-output-directory %s ' \
                      '--seed %d --time-limit %d --per-run-time-limit %d ' \
                      '--ml-memory-limit %d --resampling-strategy partial-cv ' \
                      '--folds 10 --ensemble-size 0 ' \
                      '--metalearning-configurations 0 ' \
                      '--data-format automl-competition-format ' \
                      '--dataset %s' % (output_directory,
                                        output_directory,
                                        num_run * 1000,
                                        args.time_limit,
                                        args.per_run_time_limit,
                                        args.ml_memory_limit, dataset)
            commands.append(command)

    commands_file = os.path.join(args.output_directory, 'commands.txt')
    with open(commands_file, 'w') as fh:
        for command in commands:
            fh.write("%s\n" % command)