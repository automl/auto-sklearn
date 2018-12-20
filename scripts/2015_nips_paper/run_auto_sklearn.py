import argparse
import score_vanilla
import score_ensemble


def main(working_directory, output_file, task_id, seed, model, time_limit, per_run_time_limit):
    # calls one of score_vanilla, score_ens, score_meta, score_ensmeta with given task-id, seed.
    if model == "vanilla":
        # run vanilla as on given task id and seed and store results.
        score_vanilla.main(working_directory,
                        time_limit,
                        per_run_time_limit,
                        task_id,
                        seed,
                        )
    elif model == "ensemble":
        # run ens as on given task id and seed and store results.
        score_ensemble.main(working_directory,
                            output_file,
                            task_id,
                            seed,
                            ensemble_size=50,
                            )
    elif model == "metalearning":
        # run meta as on given task id and seed and store results.
        pass
    elif model == "meta_ensemble":
        # run metaens as on given task id and seed and store results.
        pass


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--working-directory', type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('-s', '--seed', type=int)
    #parser.add_argument("--ensemble-size", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--model", type=str, required=True) ## one of (vanilla, ensemble, metalearning, meta_ensemble)

    args = parser.parse_args()
    working_directory = args.working_directory
    output_file = args.output_file
    task_id = args.task_id
    seed = args.seed
    n_jobs = args.n_jobs
    model = args.model
    # Time to run experiment
    time_limit = 50
    per_run_time_limit = 17

    main(working_directory, output_file, task_id, seed, model, time_limit, per_run_time_limit)
