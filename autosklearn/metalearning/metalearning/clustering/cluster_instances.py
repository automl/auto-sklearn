import argparse
from collections import deque
import logging
import time

import numpy as np
import scipy.stats
import sklearn.cluster
import sklearn.manifold
import sklearn.preprocessing
import sklearn.utils

from pyMetaLearn.metalearning.meta_base import MetaBase, Run
from ConfigSpace.configuration_space import ConfigurationSpace

logging.basicConfig()


parser = argparse.ArgumentParser()
parser.add_argument("metadata_dir", type=str)
args = parser.parse_args()

metadata_dir = args.metadata_dir

configuration_space = ConfigurationSpace()

meta_base = MetaBase(configuration_space, args.metadata_dir)
metafeatures = meta_base.metafeatures.copy()
metafeatures = metafeatures.fillna(metafeatures.mean())

scaler = sklearn.preprocessing.MinMaxScaler()
metafeatures.values[:,:] = scaler.fit_transform(metafeatures.values)
