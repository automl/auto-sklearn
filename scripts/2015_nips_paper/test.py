import os
import glob

working_directory = "log_output"

vanilla_seed_dir = os.path.join(working_directory, 'vanilla')
seed_list = [seed for seed in os.listdir(vanilla_seed_dir)]
#print(seed_list)

vanilla_task_dir = os.path.join(vanilla_seed_dir, seed_list[0])
task_list = [task_id for task_id in os.listdir(vanilla_task_dir)]

for model in ['vanilla', 'ensemble', 'metalearning', 'meta_ens']:
    for task_id in task_list:
        csv_files = []

        for seed in seed_list:
            # Handling the two cases separately here because they are located in different folders.
            if model in ['vanilla', 'ensemble']:
                # no metalearning (vanilla, ensemble)
                csv_file = os.path.join(working_directory,
                                                        'vanilla',
                                                        seed,
                                                        task_id,
                                                        "score_{}.csv".format(model)
                                                        )
                csv_files.append(csv_file)

            elif model in ['metalearning', 'meta_ens']:
                # Metalearning (metalearning, meta_ensemble)
                csv_file = os.path.join(working_directory,
                                                 'metalearning',
                                                 seed,
                                                 task_id,
                                                 "score_{}.csv".format(model)
                                                 )
                csv_files.append(csv_file)

        print(csv_files)
