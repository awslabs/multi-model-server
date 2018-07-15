# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Logging utilities.
"""
import logging
import sys

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


def get_logger(name=None, level="NOTSET"):
    """Gets a customized logger.

    Parameters
    ----------
    name: str, optional
        Name of the logger.
    level: int, optional
        The `logging` level for the logger.
        See: https://docs.python.org/2/library/logging.html#logging-levels

    Returns
    -------
    Logger
        A customized `Logger` object.

    """

    logger = logging.getLogger(name)
    if name is not None and not getattr(logger, '_init_done', None):
        logger._init_done = True
        hdlr = logging.StreamHandler(sys.stdout)
        # the `_Formatter` contain some escape character to
        # represent color, which is not suitable for FileHandler,
        hdlr.setFormatter(_Formatter())
        logger.addHandler(hdlr)
        logger.setLevel(LOG_LEVEL_DICT[level])
    return logger


def log_msg(*args):
    msg = " ".join(a for a in args)
    sys.stdout.write(msg)
    sys.stdout.flush()


def log_error(*args):
    msg = " ".join(a for a in args)
    sys.stderr.write(msg)
    sys.stderr.flush()
