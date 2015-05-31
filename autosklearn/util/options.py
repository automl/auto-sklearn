from autosklearn.external.configargparse import ArgParser


def get_options():
    parser = ArgParser()
    parser.add_argument("-c", "--config", help="Path to a configuration file.",
                        is_config_file=True)

    # Arguments concerning the target dataset
    parser.add_argument("--dataset-name", type=str,
                        help="Name of the target dataset.")
    parser.add_argument("--data-dir", type=str,
                        help="Directory where the dataset resides.")

    # Arguments concerning the file output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="AutoSklearn output directory. If not specified, "
                             "a new directory under /tmp/ will be generated.")
    parser.add_argument("--temporary-output-directory", type=str,
                        help="Temporary output directory. If not specified, "
                             "a new directory under /tmp/ will be generated.",
                        default=None)
    parser.add_argument("--keep-output", action='store_true', default=False,
                        help="If output_dir and temporary_output_dir are not "
                             "specified, setting this to False will make "
                             "autosklearn not delete these two directories.")

    # Arguments concerning the configuration procedure
    parser.add_argument("--time-limit", type=int, default=3600,
                        help="Total runtime of the AutoSklearn package in "
                             "seconds.")
    parser.add_argument("--per-run-time-limit", type=int, default=360,
                        help="Runtime for a single call of a target algorithm.")
    parser.add_argument("--ml-memory-limit", type=int, default=3000,
                        help="Memory limit for the machine learning pipeline "
                             "being optimized")

    # Arguments concerning the metalearning part
    parser.add_argument("--metalearning-configurations", type=int, default=25,
                        help="Number of configurations which will be used as "
                             "initial challengers for SMAC.")

    # Arguments concerning the algorithm selection part
    parser.add_argument("--ensemble-size", type=int, default=1,
                        help="Maximum number of models in the ensemble. Set "
                             "this to one in order to evaluate the single "
                             "best.")
    parser.add_argument("--ensemble-nbest", type=int, default=1,
                        help="Consider only the best n models in the ensemble "
                             "building process.")

    # Other
    parser.add_argument("-s", "--seed", type=int, default=1,
                        help="Seed for random number generators.")
    parser.add_argument('--exec-dir', type=str,
                        help="Execution directory.")
    parser.add_argument("--metadata-directory",
                        help="DO NOT CHANGE THIS UNLESS YOU KNOW WHAT YOU ARE DOING."
                        "\nReads the metadata from a different directory. "
                        "This feature should only be used to perform studies "
                        "on the performance of AutoSklearn.")
    return parser