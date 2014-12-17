'''
Created on Dec 16, 2014

@author: Aaron Klein

     Simple script to write an AutoSklearn configuration space for a classifier to a .pcs file
'''


from AutoSklearn.autosklearn import AutoSklearnClassifier
from HPOlibConfigSpace.converters import pcs_parser

cs = AutoSklearnClassifier.get_hyperparameter_search_space()

with open("params.pcs", 'w') as fh:
    fh.write(pcs_parser.write(cs))
