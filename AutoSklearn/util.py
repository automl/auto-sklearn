import importlib
import inspect
import os
import pkgutil
import sklearn
import sklearn.base
import sys


class NoModelException(Exception):
    def __init__(self, cls, method):
        self.cls = cls
        self.method = method

    def __str__(self):
        return repr("You called %s.%s without specifying a model first."
                    % (type(self.cls), self.method))


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

find_sklearn_classifiers()