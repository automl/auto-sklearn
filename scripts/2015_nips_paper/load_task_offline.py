import io
import os
import xmltodict
import numpy as np

import openml
import openml.tasks.functions as t
import openml.datasets.functions as d

def get_dataset_offline(dataset_id):
    # Workaround for offline
    cache_path = os.path.join(os.path.expanduser("~"), "openml")
    #cache_path = os.path.expanduser("~")

    # Description for dataset
    description_file = os.path.join(cache_path, "datasets/{}/description.xml".format(dataset_id))
    with io.open(description_file, encoding='utf8') as fh:
        dataset_xml = fh.read()

    # features for dataset
    features_file = os.path.join(cache_path, "datasets/{}/features.xml".format(dataset_id))
    with io.open(features_file, encoding='utf8') as fh:
        features_xml = fh.read()

    # qualities for dataset
    qualities_file = os.path.join(cache_path, "datasets/{}/qualities.xml".format(dataset_id))
    with io.open(qualities_file, encoding='utf8') as fh:
        qualities_xml = fh.read()
        
    # Arff file for dataset
    arff = os.path.join(cache_path, "datasets/{}/dataset.arff".format(dataset_id))
    with io.open(arff, encoding='utf8'):
        pass

    description = xmltodict.parse(dataset_xml)["oml:data_set_description"]
    features = xmltodict.parse(features_xml)["oml:data_features"]
    qualities = xmltodict.parse(qualities_xml)["oml:data_qualities"]['oml:quality']
    arff_file = arff
    dataset = d._create_dataset_from_description(description, features, qualities, arff_file)

    return dataset
    
def get_task_offline(task_id):
    # Workaround for offline
    cache_path = os.path.join(os.path.expanduser("~"), "openml")
    #cache_path = os.path.expanduser("~")

    task_file = os.path.join(cache_path, "tasks/{}/task.xml".format(task_id))

    with io.open(task_file, encoding='utf8') as fh:
        task = t._create_task_from_xml(xml=fh.read())

    return task


def load_task(task_id):
    # Create task and dataset object offline.
    task = get_task_offline(task_id)
    dataset = get_dataset_offline(task.dataset_id)

    # Get X and y.
    X, y, cat = dataset.get_data(return_categorical_indicator=True,
                             target=task.target_name)
    cat = ['categorical' if c else 'numerical' for c in cat]

    # Train test split.
    train_indices, test_indices = task.get_train_test_split_indices()
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    unique = np.unique(y_train)
    mapping = {unique_value: i for i, unique_value in enumerate(unique)}
    y_train = np.array([mapping[value] for value in y_train])
    y_test = np.array([mapping[value] for value in y_test])

    return X_train, y_train, X_test, y_test, cat
