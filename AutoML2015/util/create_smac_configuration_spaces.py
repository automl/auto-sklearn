'''
Created on Dec 16, 2014

@author: Aaron Klein
@author: Matthias Feurer
'''

import os
import sys

from models.autosklearn import get_configuration_space
from data.data_io import inventory_data
from data.data_manager import DataManager
from HPOlibConfigSpace.converters import pcs_parser


input_dir = sys.argv[1]
output_dir = sys.argv[2]
datanames = inventory_data(input_dir)

try:
    os.mkdir(output_dir)
except:
    pass

for basename in datanames:
    D = DataManager(basename, input_dir, verbose=True)
    if D.info['task'] == 'regression':
        # Not yet implemented...
        continue

    cs = get_configuration_space(D.info)
    dataset_dir = os.path.join(output_dir, basename)

    try:
        os.mkdir(dataset_dir)
    except:
        pass

    with open(os.path.join(dataset_dir, "params.pcs"), 'w') as fh:
        fh.write(pcs_parser.write(cs))
    print
