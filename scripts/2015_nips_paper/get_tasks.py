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

def get_tasks_ids(dataset_ids):
    # return task ids of corresponding datset ids.
    task_ids = []

    # Some datasets are deactivated. We remove them from the list first.
    deactivated_list = openml.datasets.list_datasets(status='deactivated')
    deactivated_list = list(deactivated_list.keys())
    for idx in deactivated_list:
        if idx in dataset_ids:
            #print("datset id %i is deactivated. Removing from list" % idx)
            dataset_ids.remove(idx)

    # Get task ids.
    tasks = openml.tasks.list_tasks(task_type_id=1)
    tasks = pd.DataFrame.from_dict(tasks, orient="index")

    for did in dataset_ids:
        task = tasks.query('did == {}'.format(did))  # find all tasks with given dataset id.
        task = list(task.tid)
        task_ids.append(task[0])  # append the first task id found.

    return task_ids

def main():
    datasets = 'resources/datasets.csv'
    dataset_ids = read_csv(datasets)
    task_ids = sorted(get_tasks_ids(dataset_ids))
    string_to_print = ''
    for tid in task_ids:
        string_to_print += str(tid) + ' '
    print(string_to_print)



if __name__=="__main__":
    main()


