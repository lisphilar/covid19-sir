#!/usr/bin/env python
# -*- coding: utf-8 -*-

from inspect import getmembers, ismethod
import sys
from loguru import logger as loguru_logger


def _catch_exceptions(cls):
    """Catch exceptions raised by the all methods of the class in logger.
    Args:
        cls (object): class object

    Returns:
        object
    """
    for name, fn in getmembers(cls):
        if isinstance(fn, (classmethod, staticmethod)):
            setattr(cls, name, type(fn)(loguru_logger.catch(fn)))
        elif ismethod(fn):
            setattr(cls, name, loguru_logger.catch(fn))
    return cls


class _ExceptionCatcher(type):
    """Basic class for catch exceptions with logger.
    """

    def __new__(cls, *args, **kwargs):
        cls = super().__new__(cls, *args, **kwargs)
        return _catch_exceptions(cls)


class _BaseClass(metaclass=_ExceptionCatcher):
    pass


class _Config(_BaseClass):
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
        try:
            log_level = self._LEVEL_DICT[level]
        except KeyError:
            candidates_str = ', '.join([str(k) for k in self._LEVEL_DICT.keys()])
            self._logger.exception(
                f"@level must be selected from [{candidates_str}], but {level} was applied.")
            raise
        self._logger.remove()
        self._logger.add(
            sys.__stdout__,
            level=log_level,
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


config = _Config()
config.logger(level=2)
