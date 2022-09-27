#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from loguru import logger as loguru_logger


def _catch_exception(cls):
    """Catch exceptions raised by the all methods of the class in logger.

    Args:
        cls (object): class object
    """
    for attr in cls.__dict__:
        if callable(getattr(cls, attr)):
            setattr(cls, attr, loguru_logger.catch(getattr(cls, attr)))
    return cls


class _Config(object):
    """Class for configuration of logger.
    """
    _LEVEL_DICT = {0: "ERROR", 1: "WARNING", 2: "INFO", 3: "DEBUG"}

    def __init__(self):
        self._logger = loguru_logger

    def logger(self, level):
        """Update configuration of logger.

        Args:
            level (int): log level (0: ERROR, 1: WARNING, 2: INFO, 3: DEBUG)
        """
        self._logger.remove()
        self._logger.add(
            sys.__stdout__,
            level=self._LEVEL_DICT[level],
            format="{time:YYYY-MM-DD at HH:mm:ss} | <level>{level}</level> | <level>{message}</level>",
        )

    def warning(self, message):
        """Show warning.

        Args:
            message (str): message to show
        """
        self._logger.warning(message)

    def info(self, message):
        """Show information.

        Args:
            message (str): message to show
        """
        self._logger.info(message)

    def debug(self, message):
        """Show debug message.

        Args:
            message (str): message to show
        """
        self._logger.debug(message)

    def exception(self, **kwargs):
        """Show message for trace.

        Args:
            **kwarg: keyword arguments of the exception object

        Examples:
            >>> import covsirphy as cs
            >>> try:
                    a = "value"
                    cs.Validator(a).int()
                except cs.UnExpectedTypeError:
                    cs.config.exception(name="a", target=a, expected=int)

        """
        _exception, *_ = sys.exc_info()
        self._logger.opt(exception=True).debug(f"{_exception} was raised")
        self._logger.exception(_exception(**kwargs))


config = _Config()
config.logger(level=2)
