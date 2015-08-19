# -*- encoding: utf-8 -*-
import logging
import os
import sys


def setup():
    logging.basicConfig(level=logging.DEBUG)


def get_logger(name, outputdir=None):
    # Root logger with a stream and file handler
    root = logging.getLogger()
    formatter = logging.Formatter(fmt='[%(levelname)s] '
                                  '[%(asctime)s:%(name)s]: %(message)s',
                                  datefmt='%m-%d %H:%M:%S')

    if not any([isinstance(handler, logging.StreamHandler)
                for handler in root.handlers]):
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        root.addHandler(console)

    if outputdir is not None:
        logger_file = os.path.join(outputdir, '%s.log' % str(name))

        add = True
        for handler in root.handlers:
            if isinstance(handler, logging.FileHandler):
                if handler.baseFilename == logger_file:
                    add = False

        if add:
            file_handler = logging.FileHandler(filename=logger_file, mode='w')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.debug('Logger created')

    return logger
