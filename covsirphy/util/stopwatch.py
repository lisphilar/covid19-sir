#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime


class StopWatch(object):
    """
    Calculate elapsed time.
    """
    def __init__(self):
        self.start_time = datetime.now()
        self.elapsed = 0

    def stop(self):
        """
        Stop the stopwatch.
        @return <int>: elapsed time [sec]
        """
        end_time = datetime.now()
        self.elapsed = (end_time - self.start_time).total_seconds()
        return self.elapsed

    def show(self):
        """
        Show the elapsed time as string.
        @return <str>: eg. '1 min 30 sec'
        """
        minutes, seconds = divmod(int(self.elapsed), 60)
        return f"{minutes} min {seconds} sec"
