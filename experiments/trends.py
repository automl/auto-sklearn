from pprint import pprint
import sklearn.metrics
import autosklearn.classification
import timeit
import pandas as pd
from sklearn.model_selection import train_test_split


file_path = '../data/titanic_dirty_data.csv'
df = pd.read_csv(file_path)
df = df.dropna(subset=['Survived'])
y = df['Survived']
X = df.drop('Survived', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

time_budget = [30] #, 40, 60, 90, 120]
accuracy = []
precision = []
recall = []
leaderboard = []
config_list = []
for t in time_budget:
    print("Time budget:", t)
    automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=t,
    include = {
        'classifier': ["mlp"],
        'feature_preprocessor': ["no_preprocessing"]
    },
    tmp_folder="tmp/autosklearn_classification_example_tmp2",
    )

    automl.fit(X_train, y_train, dataset_name="airbnb")

    run_key = list(automl.automl_.runhistory_.data.keys())[0]
    run_value = automl.automl_.runhistory_.data[run_key]
    config=automl.automl_.runhistory_.ids_config[run_key.config_id]
    print("Config:", config)
    print("Leaderboard:", automl.leaderboard())
    config_list.append(str(config))
    leaderboard.append(str(automl.leaderboard()))

    predictions = automl.predict(X_test)
    # print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
    # print("Precision", sklearn.metrics.precision_score(y_test, predictions))
    # print("Recall", sklearn.metrics.recall_score(y_test, predictions))
    accuracy.append(sklearn.metrics.accuracy_score(y_test, predictions))
    precision.append(sklearn.metrics.precision_score(y_test, predictions))
    recall.append(sklearn.metrics.recall_score(y_test, predictions))

