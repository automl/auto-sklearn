import logging
import os
import sys


def get_logger(name, outputdir=None):
    # Creates a logger with a stream and file handler
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt='real_format: [%(levelname)s] '
                                      '[%(asctime)s:%(name)s]: %(message)s',
                                  datefmt='%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logger.addHandler(console)

    if outputdir is not None:
        logger_file = os.path.join(outputdir, '%s.log' % name)
        file_handler = logging.FileHandler(filename=logger_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.debug("Logger created")

    return logger
