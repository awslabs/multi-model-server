# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

#!/usr/bin/env python

"""Logging utilities."""
import logging
import sys
import warnings

LOG_LEVEL_DICT = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET
}

PY3 = sys.version_info[0] == 3


class _Formatter(logging.Formatter):
    """Customized log formatter."""

    def __init__(self, colored=True):
        self.colored = colored
        super(_Formatter, self).__init__()

    def _get_color(self, level):
        if logging.WARNING <= level:
            return '\x1b[31m'
        elif logging.INFO <= level:
            return '\x1b[32m'
        return '\x1b[34m'

    def format(self, record):
        fmt = ''
        if self.colored:
            fmt = self._get_color(record.levelno)
        fmt += '[' + logging.getLevelName(record.levelno)
        fmt += ' %(asctime)s PID:%(process)d %(pathname)s:%(funcName)s:%(lineno)d'
        if self.colored:
            fmt += ']\x1b[0m'
        fmt += ' %(message)s'
        if PY3:
            self._style._fmt = fmt 
        else:
            self._fmt = fmt
        return super(_Formatter, self).format(record)


def getLogger(name=None, filename=None, filemode=None, level='NOTSET'):
    """Gets a customized logger.

    .. note:: `getLogger` is deprecated. Use `get_logger` instead.

    """
    warnings.warn("getLogger is deprecated, Use get_logger instead.",
                  DeprecationWarning, stacklevel=2)
    return get_logger(name, filename, filemode, LOG_LEVEL_DICT[level])


def get_logger(name=None, filename=None, level="NOTSET", rotate_value='H', rotate_interval=1):
    """Gets a customized logger.

    Parameters
    ----------
    name: str, optional
        Name of the logger.
    filename: str, optional
        The filename to which the logger's output will be sent.
    filemode: str, optional
        The file mode to open the file (corresponding to `filename`),
        default is 'a' if `filename` is not ``None``.
    level: int, optional
        The `logging` level for the logger.
        See: https://docs.python.org/2/library/logging.html#logging-levels

    Returns
    -------
    Logger
        A customized `Logger` object.

    Example
    -------
    ## get_logger call with default parameters.
    >>> from mxnet.log import get_logger
    >>> logger = get_logger("Test")
    >>> logger.warn("Hello World")
    W0505 00:29:47 3525 <stdin>:<module>:1] Hello World

    ## get_logger call with WARNING level.
    >>> import logging
    >>> logger = get_logger("Test2", level=logging.WARNING)
    >>> logger.warn("Hello World")
    W0505 00:30:50 3525 <stdin>:<module>:1] Hello World
    >>> logger.debug("Hello World") # This doesn't return anything as the level is logging.WARNING.

    ## get_logger call with DEBUG level.
    >>> logger = get_logger("Test3", level=logging.DEBUG)
    >>> logger.debug("Hello World") # Logs the debug output as the level is logging.DEBUG.
    D0505 00:31:30 3525 <stdin>:<module>:1] Hello World
    """

    logger = logging.getLogger(name)
    if name is not None and not getattr(logger, '_init_done', None):
        logger._init_done = True
        if filename:
            hdlr = logging.handlers.TimedRotatingFileHandler(filename, when=rotate_value, interval=rotate_interval)
        else:
            hdlr = logging.StreamHandler()
            # the `_Formatter` contain some escape character to
            # represent color, which is not suitable for FileHandler,
            # (TODO) maybe we can add another Formatter for FileHandler.
            hdlr.setFormatter(_Formatter())
        logger.addHandler(hdlr)
        logger.setLevel(LOG_LEVEL_DICT[level])
    return logger