#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from loguru import logger as loguru_logger


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


config = _Config()
config.logger(level=2)
