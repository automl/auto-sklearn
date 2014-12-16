'''
Created on Dec 16, 2014

@author: Aaron Klein
'''


from AutoSklearn.autosklearn import AutoSklearnClassifier
from HPOlibConfigSpace.converters import pcs_parser

cs = AutoSklearnClassifier.get_hyperparameter_search_space()
print cs
#with open("params.pcs", 'w') as fh:
#     #fh.write(pcs_parser.write(cs))
print pcs_parser.write(cs)
