# -*- encoding: utf-8 -*-
import logging
import logging.config
import os
from typing import Any, Dict, Optional

import yaml


def setup_logger(output_file: Optional[str] = None, logging_config: Optional[Dict] = None
                 ) -> None:
    # logging_config must be a dictionary object specifying the configuration
    # for the loggers to be used in auto-sklearn.
    if logging_config is not None:
        if output_file is not None:
            logging_config['handlers']['file_handler']['filename'] = output_file
        logging.config.dictConfig(logging_config)
    else:
        with open(os.path.join(os.path.dirname(__file__), 'logging.yaml'),
                  'r') as fh:
            logging_config = yaml.safe_load(fh)
        if output_file is not None:
            logging_config['handlers']['file_handler']['filename'] = output_file
        logging.config.dictConfig(logging_config)


def _create_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def get_logger(name: str) -> 'PickableLoggerAdapter':
    logger = PickableLoggerAdapter(name)
    return logger


class PickableLoggerAdapter(object):

    def __init__(self, name: str):
        self.name = name
        self.logger = _create_logger(name)

    def __getstate__(self) -> Dict[str, Any]:
        """
        Method is called when pickle dumps an object.

        Returns
        -------
        Dictionary, representing the object state to be pickled. Ignores
        the self.logger field and only returns the logger name.
        """
        return {'name': self.name}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Method is called when pickle loads an object. Retrieves the name and
        creates a logger.

        Parameters
        ----------
        state - dictionary, containing the logger name.

        """
        self.name = state['name']
        self.logger = _create_logger(self.name)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.exception(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.critical(msg, *args, **kwargs)

    def log(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.log(level, msg, *args, **kwargs)

    def isEnabledFor(self, level: int) -> bool:
        return self.logger.isEnabledFor(level)
