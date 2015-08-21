# -*- encoding: utf-8 -*-

import logging
from logging.handlers import RotatingFileHandler


def get_handler_std(level, log_format):
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=log_format,
                                           datefmt='%m-%d %H:%M:%S'))
    return handler


def get_handler(filename, level, log_format):
    handler = RotatingFileHandler(filename,
                                  maxBytes=5 * 1024 * 1024,
                                  backupCount=5)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=log_format,
                                           datefmt='%m-%d %H:%M:%S'))
    return handler


def get_log_format():
    return '[%(levelname)s] [%(asctime)s:%(name)s]: %(message)s'


def add_file_handler(logger, filepath):
    logger.addHandler(
        get_handler(filepath,
                    logging.DEBUG, get_log_format())
    )


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    logger.addHandler(
        get_handler_std(logging.DEBUG, get_log_format())
    )

    logger.debug('Logger created')
    return logger

