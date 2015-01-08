import importlib
import inspect
import os
import pkgutil

import numpy as np
import scipy.sparse
import sklearn
import sklearn.base
import sklearn.datasets


def find_sklearn_classifiers():
    classifiers = set()
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
                classifiers.add(classifier)

    print classifiers


def find_sklearn_regressor():
    classifiers = set()
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
                    issubclass(obj, sklearn.base.RegressorMixin):
                classifier = obj
                print member_name, obj
                classifiers.add(classifier)

    print classifiers


def get_dataset(dataset='iris', make_sparse=False):
    iris = getattr(sklearn.datasets, "load_%s" % dataset)()
    X = iris.data
    Y = iris.target
    rs = np.random.RandomState(42)
    indices = np.arange(X.shape[0])
    train_size = len(indices) / 3. * 2.
    rs.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train = X[:100]
    Y_train = Y[:100]
    X_test = X[100:]
    Y_test = Y[100:]

    if make_sparse:
        X_train[:,0] = 0
        X_train[np.random.random(X_train.shape) > 0.5] = 0
        X_train = scipy.sparse.csc_matrix(X_train)
        X_train.eliminate_zeros()
        X_test[:,0] = 0
        X_test[np.random.random(X_test.shape) > 0.5] = 0
        X_test = scipy.sparse.csc_matrix(X_test)
        X_test.eliminate_zeros()

    return X_train, Y_train, X_test, Y_test


def _test_classifier(Classifier, dataset='iris'):
    X_train, Y_train, X_test, Y_test = get_dataset(dataset=dataset,
                                                   make_sparse=False)
    configuration_space = Classifier.get_hyperparameter_search_space()
    default = configuration_space.get_default_configuration()
    classifier = Classifier(random_state=1,
                            **{hp.hyperparameter.name: hp.value for hp in
                             default.values.values()})
    predictor = classifier.fit(X_train, Y_train)
    predictions = predictor.predict(X_test)
    return predictions, Y_test


def _test_preprocessing(Preprocessor, dataset='iris', make_sparse=False):
    X_train, Y_train, X_test, Y_test = get_dataset(dataset=dataset,
                                                   make_sparse=make_sparse)
    original_X_train = X_train.copy()
    configuration_space = Preprocessor.get_hyperparameter_search_space()
    default = configuration_space.get_default_configuration()
    preprocessor = Preprocessor(random_state=1,
                                **{hp.hyperparameter.name: hp.value for hp in
                                default.values.values()})

    transformer = preprocessor.fit(X_train, Y_train)
    return transformer.transform(X_train), original_X_train


def _test_regressor(Regressor, dataset='diabetes'):
    X_train, Y_train, X_test, Y_test = get_dataset(dataset=dataset,
                                                   make_sparse=False)
    configuration_space = Regressor.get_hyperparameter_search_space()
    default = configuration_space.get_default_configuration()
    regressor = Regressor(random_state=1,
                          **{hp.hyperparameter.name: hp.value for hp in
                          default.values.values()})
    predictor = regressor.fit(X_train, Y_train)
    predictions = predictor.predict(X_test)
    return predictions, Y_test


if __name__ == "__main__":
    find_sklearn_classifiers()
    find_sklearn_regressor()