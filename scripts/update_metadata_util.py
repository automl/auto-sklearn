import numpy as np
import openml


classification_tasks = [233, 236, 242, 244, 246, 75090, 248, 251, 75124, 253,
                        254, 75092, 258,
                        75093, 260, 261, 262, 75095, 266, 3043, 75097, 75098,
                        75099, 75159,
                        75100, 275, 288, 75103, 75105, 75106, 75107, 75108,
                        75109, 75110,
                        75112, 75113, 75114, 75115, 75116, 75117, 75119, 75120,
                        75121, 75123,
                        252, 75125, 75126, 75127, 75128, 75129, 2117, 2119,
                        2120, 2122, 2123,
                        75096, 75132, 75133, 75139, 75141, 75142, 75143, 75146,
                        75148, 75150,
                        75153, 75154, 75230, 75156, 75157, 273, 2350, 75197,
                        75163, 75166,
                        75169, 75171, 75172, 75173, 75174, 75175, 75176, 75177,
                        75179, 75240,
                        75184, 75185, 75189, 75191, 75192, 75193, 75195, 75196,
                        75234, 75161,
                        75198, 75168, 75201, 75202, 75203, 75188, 75205, 75207,
                        75210, 75212,
                        75213, 75215, 75217, 75219, 75221, 75222, 75223, 75134,
                        75225, 75226,
                        75227, 75231, 75232, 75233, 75101, 75235, 75236, 75237,
                        75178, 75239,
                        75181, 75187, 75250, 75249, 75248, 75243, 75244, 75182]
regression_tasks = [2280, 2288, 2289, 2292, 2300, 2306, 2307, 2309, 2313,
                    2315, 4768, 4769, 4772, 4774, 4779, 4790, 4796, 4835,
                    4840, 4881, 4883, 4885, 4892, 4893, 5022, 5024, 7393]


def load_task(task_id):
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y()
    train_indices, test_indices = task.get_train_test_split_indices()
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    dataset = openml.datasets.get_dataset(task.dataset_id)
    _, _, cat, _ = dataset.get_data(target=task.target_name)
    del _
    del dataset
    cat = ['categorical' if c else 'numerical' for c in cat]

    unique = np.unique(y_train)
    mapping = {unique_value: i for i, unique_value in enumerate(unique)}
    y_train = np.array([mapping[value] for value in y_train])
    y_test = np.array([mapping[value] for value in y_test])

    return X_train, y_train, X_test, y_test, cat
