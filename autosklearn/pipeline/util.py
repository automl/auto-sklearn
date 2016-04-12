import importlib
import inspect
import os
import pkgutil
import unittest

import numpy as np
import scipy.sparse
import sklearn
import sklearn.base
import sklearn.datasets


def find_sklearn_classes(class_):
    classifiers = set()
    all_subdirectories = []
    sklearn_path = sklearn.__path__[0]
    for root, dirs, files in os.walk(sklearn_path):
        all_subdirectories.append(root)

    for module_loader, module_name, ispkg in \
            pkgutil.iter_modules(all_subdirectories):

        # Work around some issues...
        if module_name in ["hmm", "mixture"]:
            print("Skipping %s" % module_name)
            continue

        module_file = module_loader.__dict__["path"]
        sklearn_module = module_file.replace(sklearn_path, "").replace("/", ".")
        full_module_name = "sklearn" + sklearn_module + "." + module_name

        pkg = importlib.import_module(full_module_name)

        for member_name, obj in inspect.getmembers(pkg):
            if inspect.isclass(obj) and \
                    issubclass(obj, class_):
                classifier = obj
                # print member_name, obj
                classifiers.add(classifier)

    print()
    for classifier in sorted([str(cls) for cls in classifiers]):
        print(classifier)


def get_dataset(dataset='iris', make_sparse=False, add_NaNs=False,
                train_size_maximum=150, make_multilabel=False,
                make_binary=False):
    iris = getattr(sklearn.datasets, "load_%s" % dataset)()
    X = iris.data.astype(np.float32)
    Y = iris.target
    rs = np.random.RandomState(42)
    indices = np.arange(X.shape[0])
    train_size = min(int(len(indices) / 3. * 2.), train_size_maximum)
    rs.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_test = X[train_size:]
    Y_test = Y[train_size:]

    if add_NaNs:
        mask = rs.choice([True, False], size=(X_train.shape))
        X_train[mask] = np.NaN

    if make_sparse:
        X_train[:,0] = 0
        X_train[rs.random_sample(X_train.shape) > 0.5] = 0
        X_train = scipy.sparse.csc_matrix(X_train)
        X_train.eliminate_zeros()
        X_test[:,0] = 0
        X_test[rs.random_sample(X_test.shape) > 0.5] = 0
        X_test = scipy.sparse.csc_matrix(X_test)
        X_test.eliminate_zeros()

    if make_binary and make_multilabel:
        raise ValueError('Can convert dataset only to one of the two '
                         'options binary or multilabel!')

    if make_binary:
        Y_train[Y_train > 1] = 1
        Y_test[Y_test > 1] = 1

    if make_multilabel:
        num_classes = len(np.unique(Y))
        Y_train_ = np.zeros((Y_train.shape[0], num_classes))
        for i in range(Y_train.shape[0]):
            Y_train_[i, Y_train[i]] = 1
        Y_train = Y_train_
        Y_test_ = np.zeros((Y_test.shape[0], num_classes))
        for i in range(Y_test.shape[0]):
            Y_test_[i, Y_test[i]] = 1
        Y_test = Y_test_

    return X_train, Y_train, X_test, Y_test


def _test_classifier(classifier, dataset='iris', sparse=False,
                     train_size_maximum=150, make_multilabel=False,
                     make_binary=False):
    X_train, Y_train, X_test, Y_test = get_dataset(dataset=dataset,
                                                   make_sparse=sparse,
                                                   train_size_maximum=train_size_maximum,
                                                   make_multilabel=make_multilabel,
                                                   make_binary=make_binary)

    configuration_space = classifier.get_hyperparameter_search_space(
        dataset_properties={'sparse': sparse})
    default = configuration_space.get_default_configuration()
    classifier = classifier(random_state=np.random.RandomState(1),
                            **{hp_name: default[hp_name] for hp_name in
                               default if default[hp_name] is not None})
    predictor = classifier.fit(X_train, Y_train)
    predictions = predictor.predict(X_test)
    return predictions, Y_test


def _test_classifier_iterative_fit(classifier, dataset='iris', sparse=False):
    """Fit only for ten iterations. Usually, the result is much worse which
    indicates that iterative_fit() works"""
    X_train, Y_train, X_test, Y_test = get_dataset(dataset=dataset,
                                                   make_sparse=sparse)
    configuration_space = classifier.get_hyperparameter_search_space(
        dataset_properties={'sparse': sparse})
    default = configuration_space.get_default_configuration()
    classifier = classifier(random_state=np.random.RandomState(1),
                            **{hp_name: default[hp_name] for hp_name in
                               default if default[hp_name] is not None})
    for i in range(10):
        predictor = classifier.iterative_fit(X_train, Y_train)
    predictions = predictor.predict(X_test)
    return predictions, Y_test


def _test_classifier_predict_proba(classifier, dataset='iris', sparse=False,
                                   train_size_maximum=150,
                                   make_multilabel=False,
                                   make_binary=False):
    X_train, Y_train, X_test, Y_test = get_dataset(dataset=dataset,
                                                   make_sparse=sparse,
                                                   train_size_maximum=train_size_maximum,
                                                   make_multilabel=make_multilabel,
                                                   make_binary=make_binary)
    configuration_space = classifier.get_hyperparameter_search_space()
    default = configuration_space.get_default_configuration()
    classifier = classifier(random_state=np.random.RandomState(1),
                            **{hp_name: default[hp_name] for hp_name in
                               default})
    predictor = classifier.fit(X_train, Y_train)
    predictions = predictor.predict_proba(X_test)
    return predictions, Y_test


def _test_preprocessing(Preprocessor, dataset='iris', make_sparse=False):
    X_train, Y_train, X_test, Y_test = get_dataset(dataset=dataset,
                                                   make_sparse=make_sparse)
    original_X_train = X_train.copy()
    configuration_space = Preprocessor.get_hyperparameter_search_space()
    default = configuration_space.get_default_configuration()

    preprocessor = Preprocessor(random_state=np.random.RandomState(1),
                                **{hp_name: default[hp_name] for hp_name in
                                   default if default[hp_name] is not None})

    transformer = preprocessor.fit(X_train, Y_train)
    return transformer.transform(X_train), original_X_train


class PreprocessingTestCase(unittest.TestCase):
    def _test_preprocessing_dtype(self, Preprocessor, add_NaNs=False,
                                  test_sparse=True, dataset='iris'):
        # Dense
        # np.float32
        X_train, Y_train, X_test, Y_test = get_dataset(dataset, add_NaNs=add_NaNs)
        self.assertEqual(X_train.dtype, np.float32)

        configuration_space = Preprocessor.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        preprocessor = Preprocessor(random_state=np.random.RandomState(1),
                                    **{hp_name: default[hp_name] for hp_name in
                                       default})
        preprocessor.fit(X_train, Y_train)
        Xt = preprocessor.transform(X_train)
        self.assertEqual(Xt.dtype, np.float32)

        # np.float64
        X_train, Y_train, X_test, Y_test = get_dataset(dataset, add_NaNs=add_NaNs)
        X_train = X_train.astype(np.float64)
        configuration_space = Preprocessor.get_hyperparameter_search_space()
        default = configuration_space.get_default_configuration()
        preprocessor = Preprocessor(random_state=np.random.RandomState(1),
                                    **{hp_name: default[hp_name] for hp_name in
                                       default})
        preprocessor.fit(X_train, Y_train)
        Xt = preprocessor.transform(X_train)
        self.assertEqual(Xt.dtype, np.float64)

        if test_sparse is True:
            # Sparse
            # np.float32
            X_train, Y_train, X_test, Y_test = get_dataset(dataset, make_sparse=True,
                                                           add_NaNs=add_NaNs)
            self.assertEqual(X_train.dtype, np.float32)
            configuration_space = Preprocessor.get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()
            preprocessor = Preprocessor(random_state=np.random.RandomState(1),
                                        **{hp_name: default[hp_name] for hp_name
                                           in default})
            preprocessor.fit(X_train, Y_train)
            Xt = preprocessor.transform(X_train)
            self.assertEqual(Xt.dtype, np.float32)

            # np.float64
            X_train, Y_train, X_test, Y_test = get_dataset(dataset,
                                                           make_sparse=True,
                                                           add_NaNs=add_NaNs)
            X_train = X_train.astype(np.float64)
            configuration_space = Preprocessor.get_hyperparameter_search_space()
            default = configuration_space.get_default_configuration()
            preprocessor = Preprocessor(random_state=np.random.RandomState(1),
                                        **{hp_name: default[hp_name] for hp_name
                                           in default})
            preprocessor.fit(X_train, Y_train)
            Xt = preprocessor.transform(X_train)
            self.assertEqual(Xt.dtype, np.float64)


def _test_regressor(Regressor, dataset='diabetes', sparse=False):
    X_train, Y_train, X_test, Y_test = get_dataset(dataset=dataset,
                                                   make_sparse=sparse)
    configuration_space = Regressor.get_hyperparameter_search_space()
    default = configuration_space.get_default_configuration()
    regressor = Regressor(random_state=np.random.RandomState(1),
                          **{hp_name: default[hp_name] for hp_name in
                             default})
    # Dumb incomplete hacky test to check that we do not alter the data
    X_train_hash = hash(str(X_train))
    X_test_hash = hash(str(X_test))
    Y_train_hash = hash(str(Y_train))
    predictor = regressor.fit(X_train, Y_train)
    predictions = predictor.predict(X_test)
    if X_train_hash != hash(str(X_train)) or \
                    X_test_hash != hash(str(X_test)) or \
                    Y_train_hash != hash(str(Y_train)):
        raise ValueError("Model modified data")
    return predictions, Y_test


def _test_regressor_iterative_fit(Regressor, dataset='diabetes', sparse=False):
    X_train, Y_train, X_test, Y_test = get_dataset(dataset=dataset,
                                                   make_sparse=sparse)
    configuration_space = Regressor.get_hyperparameter_search_space(
        dataset_properties={'sparse': sparse})
    default = configuration_space.get_default_configuration()
    regressor = Regressor(random_state=np.random.RandomState(1),
                          **{hp_name: default[hp_name] for hp_name in
                             default})
    while not regressor.configuration_fully_fitted():
        regressor = regressor.iterative_fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return predictions, Y_test


if __name__ == "__main__":
    find_sklearn_classes(sklearn.base.ClassifierMixin)
    find_sklearn_classes(sklearn.base.RegressorMixin)
    find_sklearn_classes(sklearn.base.TransformerMixin)
