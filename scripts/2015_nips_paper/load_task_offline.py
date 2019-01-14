import os
import openml.tasks.functions as functions

def load_task(task_id):
    # find the dataset from the file system
    task_xml_path = "data/aad/openml/tasks/{}/task.xml"
    task_xml_path = os.path.abspath(task_xml_path)
    task = functions._create_task_from_xml(task_xml_path)
    print("task: ",task)

    X, y = task.get_X_and_y()
    print("X: ", X)
    print("y: ", y)
    train_indices, test_indices = task.get_train_test_split_indices()
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    return
    dataset = openml.datasets.get_dataset(task.dataset_id)
    _, _, cat = dataset.get_data(return_categorical_indicator=True,
                                 target=task.target_name)
    del _
    del dataset
    cat = ['categorical' if c else 'numerical' for c in cat]

    unique = np.unique(y_train)
    mapping = {unique_value: i for i, unique_value in enumerate(unique)}
    y_train = np.array([mapping[value] for value in y_train])
    y_test = np.array([mapping[value] for value in y_test])

    return X_train, y_train, X_test, y_test, cat

def load_task(task_id):
    task = openml.tasks.get_task(task_id)

