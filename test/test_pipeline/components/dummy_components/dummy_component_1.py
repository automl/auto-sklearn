import os
import sys

from autosklearn.pipeline.components.base import AutoSklearnClassificationAlgorithm

# Add the parent directory to the path to import the parent component
this_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(this_directory, '..'))
sys.path.append(parent_directory)


class DummyComponent1(AutoSklearnClassificationAlgorithm):
    pass
