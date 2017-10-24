import os
import glob
import inspect
import importlib

import sklearn.base

files = glob.glob(os.path.join(os.path.dirname(sklearn.__file__), "**/*.py"),
                  recursive=True)

def find_all(cls):
    found = set()
    for file in files:
        parts = file.split('/')
        parts[-1] = parts[-1].replace('.py', '')
        sklearn_dir = parts.index('sklearn')
        name = '.'.join(parts[sklearn_dir:])
        module = importlib.import_module(name)
        for member in module.__dict__.values():
            if not inspect.isclass(member):
                continue
            if issubclass(member, cls):
                found.add(member)
    print('#####')
    found = list(found)
    found.sort(key=lambda t: str(t))
    for f in found:
        print(f)
    return found

#classifiers = find_all(sklearn.base.ClassifierMixin)
#regressors = find_all(sklearn.base.RegressorMixin)
preprocs = find_all(sklearn.base.TransformerMixin)

