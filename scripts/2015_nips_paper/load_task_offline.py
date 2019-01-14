import io
import os
import openml
import openml.tasks.functions as functions

def load_task(task_id):
    # find the dataset from the file system
    xml_file = "data/aad/openml/tasks/{}/task.xml"
    xml_file = os.path.abspath(xml_file)
    task_xml = openml._api_calls._perform_api_call("task/%d" % task_id)

    with io.open(xml_file, "w", encoding='utf8') as fh:
        fh.write(task_xml)
    print(task_xml)
    task = functions._create_task_from_xml(task_xml)
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

