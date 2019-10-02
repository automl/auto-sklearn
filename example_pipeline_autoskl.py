import numpy as np

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

from autosklearn.pipeline.classification import SimpleClassificationPipeline
#from autosklearn.pipeline.classification import SimpleClassificationPipeline
from autosklearn.pipeline.numeric_classification import NumericClassificationPipeline

def create_data_set(instances=100, n_feats=10, n_categ_feats=4,
    categs_per_feat=5, missing_share=.3, n_classes=3):
    # Create a base dataset
    size = (instances, n_feats)
    data = np.random.uniform(size=size)
    # Overwrite some features to make them categorical
    categ_flag = np.random.choice(n_feats, n_categ_feats, replace=False)
    numer_flag = list(set(range(n_feats)) - set(categ_flag))
    categ_data = np.random.randint(0, categs_per_feat, size=(instances, n_categ_feats))
    data[:, categ_flag] = (categ_data * 42).astype(str)  # just to make it 'more non-numerical'
    # Add missing values
    missing_mask = np.random.uniform(size=size) < missing_share
    data[missing_mask] = None
    # Create labels
    labels = np.random.randint(0, n_classes, size=instances)
    # Feature types
    feat_type = np.array(["categorical"] * n_feats)
    feat_type[numer_flag] = "numerical"

    return data, labels, feat_type.tolist()


X, y, feat_type = create_data_set()

SCP = SimpleClassificationPipeline()
SCP.fit(X, y, feat_type=feat_type)
predictions = SCP.predict(X)
print("Accuracy score", sklearn.metrics.accuracy_score(y, predictions))




quit()

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

