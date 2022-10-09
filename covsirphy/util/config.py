#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import warnings
from loguru import logger as loguru_logger


class _Config(object):
    """Class for configuration of logger.
    """
    _LEVEL_DICT = {0: "ERROR", 1: "WARNING", 2: "INFO", 3: "DEBUG"}

    def __init__(self):
        self._logger = loguru_logger
        self._logger_level = 2

    @property
    def logger_level(self):
        """int: integer to indicate the logging level
        """
        return self._logger_level

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
        self._logger_level = level

    def error(self, message):
        """Logging raised exception.

        Args:
            message (str): message to show
        """
        self._logger.error(message, exec_info=True)

    def warning(self, message, category=None):
        """Raise warning.

        Args:
            message (str): message to show
            category (Warning or None): category of warning or None (DeprecationWarning)
        """
        self._logger.warning(message)
        warnings.warn("deprecated callable was used", category, stacklevel=2)

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
