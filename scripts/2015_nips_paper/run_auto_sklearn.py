import argparse
import run_without_metalearning
import run_with_metalearning
import score_ensemble


def main(working_directory, output_file, task_id, seed, model, time_limit, per_run_time_limit):
    # calls one of score_vanilla, score_ens, score_meta, score_ensmeta with given task-id, seed.
    if model == "vanilla":
        run_without_metalearning.main(working_directory,
                                    time_limit,
                                    per_run_time_limit,
                                    task_id,
                                    seed,
                                    )
        score_ensemble.main(working_directory,
                            output_file,
                            task_id,
                            seed,
                            ensemble_size=1,
                            )
    elif model == "metalearning":
        run_with_metalearning.main(working_directory,
                                time_limit,
                                per_run_time_limit,
                                task_id,
                                seed,
                                )
        score_ensemble.main(working_directory,
                            output_file,
                            task_id,
                            seed,
                            ensemble_size=1,
                            )
    else:
        score_ensemble.main(working_directory,
                            output_file,
                            task_id,
                            seed,
                            ensemble_size=50,
                            )

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--working-directory', type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--time-limit", type=int, required=True)
    parser.add_argument("--per-runtime-limit", type=int, required=True)
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('-s', '--seed', type=int)
    parser.add_argument("--model", type=str, required=True) ## one of (vanilla, ensemble, metalearning, meta_ensemble)

    args = parser.parse_args()
    working_directory = args.working_directory # logdir/vanilla or logdir/metalearning
    output_file = args.output_file
    task_id = args.task_id
    seed = args.seed
    model = args.model
    time_limit = args.time_limit
    per_run_time_limit = args.per_runtime_limit

    main(working_directory, output_file, task_id, seed, model, time_limit, per_run_time_limit)
