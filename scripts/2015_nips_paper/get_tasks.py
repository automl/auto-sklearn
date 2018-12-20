# get tasks from openml
import openml
import pandas as pd
from pprint import pprint

def read_csv(file):
    # reads the given csv file and returns a list containing all dataset ids used for experiment.
    dataset_ids = []
    with open(file) as f:
        for line in f:
            dataset_id, _ = line.split('/')[2].split('_')
            dataset_ids.append(int(dataset_id))

    return dataset_ids

def get_task_ids(dataset_ids):
    # return task ids of corresponding datset ids.
    # active tasks
    tasks_a = openml.tasks.list_tasks(task_type_id=1, status='active')
    tasks_a = pd.DataFrame.from_dict(tasks_a, orient="index")
    # query only those with NaN as evaluation_measures.
    tasks_a = tasks_a.query("evaluation_measures != evaluation_measures")
    # query only those with holdout as the resampling startegy.
    tasks_a = tasks_a[(tasks_a.estimation_procedure == "33% Holdout set")]

    # deactivated tasks
    tasks_d = openml.tasks.list_tasks(task_type_id=1, status='deactivated')
    tasks_d = pd.DataFrame.from_dict(tasks_d, orient="index")
    tasks_d = tasks_d.query("evaluation_measures != evaluation_measures")
    tasks_d = tasks_d[(tasks_d.estimation_procedure == "33% Holdout set")]

    task_ids = []
    for did in dataset_ids:
        task_a = list(tasks_a.query("did == {}".format(did)).tid)
        task_d = list(tasks_d.query("did == {}".format(did)).tid)
        task_ids += task_a + task_d
    #print(len(task_ids)) there are only 22 tasks!

    return task_ids  # return list of all task ids.

def main():
    datasets = 'resources/datasets.csv'
    dataset_ids = read_csv(datasets)
    task_ids = sorted(get_task_ids(dataset_ids))
    string_to_print = ''
    for tid in task_ids:
        string_to_print += str(tid) + ' '
    print(string_to_print)  # print the task ids for bash script.



if __name__=="__main__":
    main()


