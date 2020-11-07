#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime


class StopWatch(object):
    """
    Calculate elapsed time.
    """

    def __init__(self):
        self.start_time = datetime.now()

    def stop(self):
        """
        Stop.

        Returns:
            int: elapsed time [sec]
        """
        end_time = datetime.now()
        return (end_time - self.start_time).total_seconds()

    @staticmethod
    def show(time_sec):
        """
        Show the elapsed time as string.

        Args:
            time_sec (int): time [sec]

        Returns:
            str: eg. '1 min 30 sec'
        """
        minutes, seconds = divmod(int(time_sec), 60)
        return f"{minutes} min {seconds:>2} sec"

    def stop_show(self):
        """
        Stop and show time.

        Returns:
            str: eg. '1 min 30 sec'
        """
        return self.show(self.stop())
