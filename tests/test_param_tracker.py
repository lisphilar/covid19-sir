#!/usr/bin/env python
# -*- coding: utf-8 -*-


import warnings
import pytest


class TestParamTracker(object):
    def test_trend(self, param_tracker):
        warnings.simplefilter("ignore", category=UserWarning)
        param_tracker.trend(show_figure=True)

    def test_trend_find_phase(self, param_tracker):
        # S-R trend analysis
        with pytest.raises(ValueError):
            param_tracker.find_phase("01May2020")
        param_tracker.trend(show_figure=False)
        # Find phase name
        with pytest.raises(IndexError):
            param_tracker.find_phase("01Jan2000")
        param_tracker.find_phase("01May2020")

    @pytest.mark.parametrize(
        "date", ["01Apr2020", "01May2020", "01Jun2020", "01Jul2020"])
    def test_separate(self, param_tracker, date):
        param_tracker.trend(show_figure=False)
        close_dates = param_tracker.near_change_dates()
        if date in close_dates:
            with pytest.raises(ValueError):
                param_tracker.separate(date)
        else:
            param_tracker.separate(date)
