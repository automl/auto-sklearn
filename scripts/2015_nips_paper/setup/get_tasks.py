# get tasks from openml
import openml
import pandas as pd


# List of dataset IDs used for the NIPS experiment.
dataset_ids = [1000, 1002, 1018, 1019, 1020, 1021, 1036, 1040, 1041, 1049, 1050, 1053,
               1056, 1067, 1068, 1069, 1111, 1112, 1114, 1116, 1119, 1120, 1128, 1130,
               1134, 1138, 1139, 1142, 1146, 1161, 1166, 12, 14, 16, 179, 180, 181, 182,
               184, 185, 18, 21, 22, 23, 24, 26, 273, 28, 293, 300, 30, 31, 32, 351, 354,
               357, 36, 389, 38, 390, 391, 392, 393, 395, 396, 398, 399, 3, 401, 44, 46,
               554, 57, 60, 679, 6, 715, 718, 720, 722, 723, 727, 728, 734, 735, 737,
               740, 741, 743, 751, 752, 761, 772, 797, 799, 803, 806, 807, 813, 816, 819,
               821, 822, 823, 833, 837, 843, 845, 846, 847, 849, 866, 871, 881, 897, 901,
               903, 904, 910, 912, 913, 914, 917, 923, 930, 934, 953, 958, 959, 962, 966,
               971, 976, 977, 978, 979, 980, 991, 993, 995]


def get_task_ids(dataset_ids):
    # return task ids of corresponding datset ids.

    # active tasks
    tasks_a = openml.tasks.list_tasks(task_type_id=1, status='active')
    tasks_a = pd.DataFrame.from_dict(tasks_a, orient="index")

    # query only those with holdout as the resampling startegy.
    tasks_a = tasks_a[(tasks_a.estimation_procedure == "33% Holdout set")]

    # deactivated tasks
    tasks_d = openml.tasks.list_tasks(task_type_id=1, status='deactivated')
    tasks_d = pd.DataFrame.from_dict(tasks_d, orient="index")

    tasks_d = tasks_d[(tasks_d.estimation_procedure == "33% Holdout set")]

    task_ids = []
    for did in dataset_ids:
        task_a = list(tasks_a.query("did == {}".format(did)).tid)
        if len(task_a) > 1:  # if there are more than one task, take the lowest one.
            task_a = [min(task_a)]
        task_d = list(tasks_d.query("did == {}".format(did)).tid)
        if len(task_d) > 1:
            task_d = [min(task_d)]
        task_ids += list(task_a + task_d)

    return task_ids  # return list of all task ids.


def main():
    task_ids = sorted(get_task_ids(dataset_ids))
    string_to_print = ''
    for tid in task_ids:
        string_to_print += str(tid) + ' '
    print(string_to_print)  # print the task ids for bash script.


if __name__ == "__main__":
    main()
