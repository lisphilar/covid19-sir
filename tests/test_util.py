#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import pandas as pd
import pytest
from covsirphy import find_args, save_dataframe, Filer, StopWatch


class TestArgument(object):
    def test_find_args(self):
        assert find_args(Filer, directory="output") == {"directory": "output"}
        assert find_args([Filer, Filer.files], directory="output") == {"directory": "output"}


class TestFiler(object):
    def test_filing(self):
        with pytest.raises(ValueError):
            Filer("output", numbering="xxx")
        filer = Filer(directory="output", prefix="jpn", suffix=None, numbering="01")
        # Create filenames
        filer.png("records")
        filer.jpg("records")
        filer.csv("records", index=True)
        # Check files
        assert len(filer.files(ext=None)) == 3
        assert len(filer.files(ext="png")) == 1
        assert len(filer.files(ext="jpg")) == 1
        assert len(filer.files(ext="csv")) == 1
        # Save CSV file
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        save_dataframe(pd.DataFrame(), filename=None, index=False)


class TestStopWatch(object):
    def test_stopwatch(self):
        stop_watch = StopWatch()
        assert isinstance(stop_watch.stop_show(), str)
