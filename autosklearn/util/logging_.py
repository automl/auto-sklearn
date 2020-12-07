# -*- encoding: utf-8 -*-
import logging
import logging.config
import logging.handlers
import multiprocessing
import os
import pickle
import random
import select
import socketserver
import struct
import threading
from typing import Any, Dict, Optional, Type

import yaml


def setup_logger(
    output_dir: str,
    filename: Optional[str] = None,
    distributedlog_filename: Optional[str] = None,
    logging_config: Optional[Dict] = None,
) -> None:
    # logging_config must be a dictionary object specifying the configuration
    # for the loggers to be used in auto-sklearn.
    if logging_config is None:
        with open(os.path.join(os.path.dirname(__file__), 'logging.yaml'), 'r') as fh:
            logging_config = yaml.safe_load(fh)

    if filename is None:
        filename = logging_config['handlers']['file_handler']['filename']
    logging_config['handlers']['file_handler']['filename'] = os.path.join(
        output_dir, filename
    )

    if distributedlog_filename is None:
        distributedlog_filename = logging_config['handlers']['distributed_logfile']['filename']
    logging_config['handlers']['distributed_logfile']['filename'] = os.path.join(
            output_dir, distributedlog_filename
        )
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


def get_named_client_logger(
    output_dir: str,
    name: str,
    host: str = 'localhost',
    port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
) -> 'PicklableClientLogger':
    logger = PicklableClientLogger(
        output_dir=output_dir,
        name=name,
        host=host,
        port=port
    )
    return logger


def _get_named_client_logger(
    output_dir: str,
    name: str,
    host: str = 'localhost',
    port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
) -> logging.Logger:
    """
    When working with a logging server, clients are expected to create a logger using
    this method. For example, the automl object will create a master that awaits
    for records sent through tcp to localhost.

    Ensemble builder will then instantiate a logger object that will submit records
    via a socket handler to the server.

    We do not need to use any format as the server will render the msg as it
    needs to.

    Parameters
    ----------
        outputdir: (str)
            The path where the log files are going to be dumped
        name: (str)
            the name of the logger, used to tag the messages in the main log
        host: (str)
            Address of where the server is gonna look for messages

    Returns
    -------
        local_loger: a logger object that has a socket handler
    """
    # Setup the logger configuration
    setup_logger(output_dir=output_dir)

    local_logger = _create_logger(name)

    # Remove any handler, so that the server handles
    # how to process the message
    local_logger.handlers.clear()

    socketHandler = logging.handlers.SocketHandler(host, port)
    local_logger.addHandler(socketHandler)

    return local_logger


class PicklableClientLogger(PickableLoggerAdapter):

    def __init__(self, output_dir: str, name: str, host: str, port: int):
        self.output_dir = output_dir
        self.name = name
        self.host = host
        self.port = port
        self.logger = _get_named_client_logger(
            output_dir=output_dir,
            name=name,
            host=host,
            port=port
        )

    def __getstate__(self) -> Dict[str, Any]:
        """
        Method is called when pickle dumps an object.

        Returns
        -------
        Dictionary, representing the object state to be pickled. Ignores
        the self.logger field and only returns the logger name.
        """
        return {
            'name': self.name,
            'host': self.host,
            'port': self.port,
            'output_dir': self.output_dir,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Method is called when pickle loads an object. Retrieves the name and
        creates a logger.

        Parameters
        ----------
        state - dictionary, containing the logger name.

        """
        self.name = state['name']
        self.host = state['host']
        self.port = state['port']
        self.output_dir = state['output_dir']
        self.logger = _get_named_client_logger(
            name=self.name,
            host=self.host,
            port=self.port,
            output_dir=self.output_dir,
        )


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    """Handler for a streaming logging request.

    This basically logs the record using whatever logging policy is
    configured locally.
    """

    def handle(self) -> None:
        """
        Handle multiple requests - each expected to be a 4-byte length,
        followed by the LogRecord in pickle format. Logs the record
        according to whatever policy is configured locally.
        """
        while True:
            chunk = self.connection.recv(4)  # type: ignore[attr-defined]
            if len(chunk) < 4:
                break
            slen = struct.unpack('>L', chunk)[0]
            chunk = self.connection.recv(slen)  # type: ignore[attr-defined]
            while len(chunk) < slen:
                chunk = chunk + self.connection.recv(slen - len(chunk))  # type: ignore[attr-defined]  # noqa: E501
            obj = self.unPickle(chunk)
            record = logging.makeLogRecord(obj)
            self.handleLogRecord(record)

    def unPickle(self, data: Any) -> Any:
        return pickle.loads(data)

    def handleLogRecord(self, record: logging.LogRecord) -> None:
        # logname is define in LogRecordSocketReceiver
        # Yet Mypy Cannot see this. This is needed so that we can
        # re-use the logging setup for autosklearn into the recieved
        # records
        if self.server.logname is not None:  # type: ignore  # noqa
            name = self.server.logname  # type: ignore  # noqa
        else:
            name = record.name
        logger = logging.getLogger(name)
        # N.B. EVERY record gets logged. This is because Logger.handle
        # is normally called AFTER logger-level filtering. If you want
        # to do filtering, do it at the client end to save wasting
        # cycles and network bandwidth!
        logger.handle(record)


def start_log_server(
    host: str,
    logname: str,
    event: threading.Event,
    port: multiprocessing.Value,
    filename: str,
    logging_config: Dict,
    output_dir: str,
) -> None:
    setup_logger(filename=filename,
                 logging_config=logging_config,
                 output_dir=output_dir)

    while True:
        # Loop until we find a valid port
        _port = random.randint(10000, 65535)
        try:
            receiver = LogRecordSocketReceiver(
                host=host,
                port=_port,
                logname=logname,
                event=event,
            )
            with port.get_lock():
                port.value = _port
            receiver.serve_until_stopped()
            break
        except OSError:
            continue


class LogRecordSocketReceiver(socketserver.ThreadingTCPServer):
    """
    This class implement a entity that receives tcp messages on a given address
    For further information, please check
    https://docs.python.org/3/howto/logging-cookbook.html#configuration-server-example
    """

    allow_reuse_address = True

    def __init__(
        self,
        host: str = 'localhost',
        port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        handler: Type[LogRecordStreamHandler] = LogRecordStreamHandler,
        logname: Optional[str] = None,
        event: threading.Event = None,
    ):
        socketserver.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.timeout = 1
        self.logname = logname
        self.event = event

    def serve_until_stopped(self) -> None:
        while True:
            rd, wr, ex = select.select([self.socket.fileno()],
                                       [], [],
                                       self.timeout)
            if rd:
                self.handle_request()
            if self.event is not None and self.event.is_set():
                break
