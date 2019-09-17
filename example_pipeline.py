import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from autosklearn.pipeline.classification import SimpleClassificationPipeline
from autosklearn.pipeline.numeric_classification import NumericClassificationPipeline


X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)


SCP = SimpleClassificationPipeline()
SCP.fit(X_train, y_train)
predictions = SCP.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))

NCP = NumericClassificationPipeline()
NCP.fit(X_train, y_train)
predictions = NCP.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))

