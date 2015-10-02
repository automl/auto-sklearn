# -*- encoding: utf-8 -*-
import logging
import logging.config
import os
import sys

import yaml


def setup_logger(output_file=None):
    with open(os.path.join(os.path.dirname(__file__), 'logging.yaml'),
              'r') as fh:
        config = yaml.load(fh)
    if output_file is not None:
        config['handlers']['file_handler']['filename'] = output_file
    logging.config.dictConfig(config)


def get_logger(name):
    logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                               'message)s', datefmt='%H:%M:%S')
    return logging.getLogger(name)
