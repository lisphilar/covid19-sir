#!/usr/bin/env python
# -*- coding: utf-8 -*-

from covsirphy import StopWatch


class TestStopWatch(object):
    def test_stopwatch(self):
        stop_watch = StopWatch()
        assert isinstance(stop_watch.stop_show(), str)
