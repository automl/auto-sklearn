import importlib
import inspect
import os
import pkgutil

import numpy as np
import sklearn
import sklearn.base
import sklearn.datasets

from .autosklearn import AutoSklearnClassifier


def find_sklearn_classifiers():
    classifiers = []
    all_subdirectories = []
    sklearn_path = sklearn.__path__[0]
    for root, dirs, files in os.walk(sklearn_path):
        all_subdirectories.append(root)

    for module_loader, module_name, ispkg in \
            pkgutil.iter_modules(all_subdirectories):

        # Work around some issues...
        if module_name in ["hmm", "mixture"]:
            print "Skipping %s" % module_name
            continue

        module_file = module_loader.__dict__["path"]
        sklearn_module = module_file.replace(sklearn_path, "").replace("/", ".")
        full_module_name = "sklearn" + sklearn_module + "." + module_name

        pkg = importlib.import_module(full_module_name)

        for member_name, obj in inspect.getmembers(pkg):
            if inspect.isclass(obj) and \
                    issubclass(obj, sklearn.base.ClassifierMixin):
                classifier = obj
                print member_name, obj
                classifiers.append(classifier)

    print classifiers


def get_iris():
    iris = sklearn.datasets.load_iris()
    X = iris.data
    Y = iris.target
    rs = np.random.RandomState(42)
    indices = np.arange(X.shape[0])
    rs.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train = X[:100]
    Y_train = Y[:100]
    X_test = X[100:]
    Y_test = Y[100:]
    return X_train, Y_train, X_test, Y_test


def test_classifier_with_iris(Classifier):
    X_train, Y_train, X_test, Y_test = get_iris()
    configuration_space = Classifier.get_hyperparameter_search_space()
    default = configuration_space.get_default_configuration()
    classifier = Classifier(random_state=1,
                            **{hp.hyperparameter.name: hp.value for hp in
                             default.values.values()})
    predictor = classifier.fit(X_train, Y_train)
    predictions = predictor.predict(X_test)
    return predictions, Y_test


def test_preprocessing_with_iris(Preprocessor):
    X_train, Y_train, X_test, Y_test = get_iris()
    configuration_space = Preprocessor.get_hyperparameter_search_space()
    default = configuration_space.get_default_configuration()
    preprocessor = Preprocessor(random_state=1,
                                **{hp.hyperparameter.name: hp.value for hp in
                                default.values.values()})
    transformer = preprocessor.fit(X_train, Y_train)
    return transformer.transform(X_test), X_test


if __name__ == "__main__":
    find_sklearn_classifiers()